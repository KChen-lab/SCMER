from typing import List, Union, Optional

import torch
from typing import Type
from ._interfaces import _ABCSelector, _ABCTorchModel
from ._base_selector import _BaseSelector
from ._owlqn import OWLQN

import warnings
import multiprocessing
import numpy as np

from sklearn.decomposition import PCA

from ._utils import TicToc, VerbosePrint
from ._umap_torch_models import _RegUmapModel, _StratifiedRegUmapModel # , _SimpleRegTsneModel


class UmapMutualEmbedding():
    def __init__(self, *, w: Union[float, str, list, np.ndarray] = 'ones',
                 n_pcs: Optional[int] = None, perplexity: float = 30.,
                 max_outer_iter: int = 5, max_inner_iter: int = 20, owlqn_history_size: int = 100,
                 eps: float = 1e-12, verbosity: int = 2, torch_precision: Union[int, str, torch.dtype] = 32,
                 torch_cdist_compute_mode: str = "use_mm_for_euclid_dist",
                 n_threads: int = 1, use_gpu: bool = False, pca_seed: int = 0, ridge: float = 0.,
                 _keep_fitting_info: bool = False):
        """
        UmapL1 model

        :param w: initial value of w, weight of each marker. Acceptable values are 'ones' (all 1),
            'uniform' (random [0, 1] values), float numbers (all set to that number),
            or a list or numpy array with specific numbers.
        :param lasso: lasso strength (i.e., strength of L1 regularization in elastic net)
        :param n_pcs: Number of PCs used to generate P matrix. Skip PCA if set to `None`.
        :param perplexity: perplexity of t-SNE modeling
        :param use_beta_in_Q: whether to use the cell specific sigma^2 calculated from P in Q. (1 / beta)
        :param max_outer_iter: number of iterations of OWL-QN
        :param max_inner_iter: number of iterations inside OWL-QN
        :param owlqn_history_size: history size for OWL-QN.
        :param eps: epsilon for considering a value to be 0.
        :param verbosity: verbosity level (0 ~ 2).
        :param torch_precision: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying. You can input 32, "32", 64, or "64".
        :param torch_cdist_compute_mode: cdist_compute_mode: compute mode for torch.cdist. By default,
            "use_mm_for_euclid_dist" to (daramatically) improve performance. However, if numerical stability became an
            issue, "donot_use_mm_for_euclid_dist" may be used instead. This option does not affect distances computed
            outside of pytorch, e.g., matrix P. Only matrix Q is affect.
        :param t_distr: By default, use t-distribution (1. / (1. + pdist2)) for Q.
            Use Normal distribution instead (exp(-pdist2)) if set to False. The latter one is not stable.
        :param n_threads: number of threads (currently only for calculating P and beta)
        :param use_gpu: whether to use GPU to train the model.
        :param pca_seed: random seed used by PCA (if applicable)
        :param ridge: ridge strength (i.e., strength of L2 regularization in elastic net)
        :param _keep_fitting_info: if `True`, write similarity matrix P to `self.P` and PyTorch model to `self.model`
        """
        self._max_outer_iter = max_outer_iter
        self._max_inner_iter = max_inner_iter
        self._owlqn_history_size = owlqn_history_size
        self._n_pcs = n_pcs
        self.w = w
        self._eps = eps
        self._perplexity = perplexity
        self._torch_precision = torch_precision
        self._torch_cdist_compute_mode = torch_cdist_compute_mode
        self._n_threads = n_threads
        self._use_gpu = use_gpu
        self._pca_seed = pca_seed
        self._ridge = ridge
        self._keep_fitting_info = _keep_fitting_info
        self.verbose_print = VerbosePrint(verbosity)


    def fit(self, X, *, P=None, beta=None):
        """
        Select markers from one dataset to keep the cell-cell similarities in the same dataset

        :param X: data matrix (cells (rows) x genes/proteins (columns))
        :param X_teacher: get target similarities from this dataset
        :param batches: (optional) batch labels
        :param P: The P matrix, if calculated in advance
        :param beta: The beta associated with P, if calculated in advance
        :param must_keep: A boolean vector indicating if a feature must be kept.
            Those features will have a fixed weight 1.
        :return:
        """
        tictoc = TicToc()
        model_class = _

        if self._n_pcs is None:
            P, beta = self._resolve_P_beta(X, P, beta, self._perplexity, tictoc, self.verbose_print.prints,
                                           self._n_threads)
        else:
            pcs = PCA(self._n_pcs, random_state=self._pca_seed).fit_transform(X_teacher)
            # print(pcs)
            P, beta = self._resolve_P_beta(pcs, P, beta, self._perplexity, tictoc, self.verbose_print.prints,
                                           self._n_threads)

        if self._keep_fitting_info:
            self.P = P

        return self._fit_core(X, P, beta, model_class, tictoc)

    @staticmethod
    def _resolve_P_beta(X, P, beta, perplexity, tictoc, print_callbacks, n_threads):
        if P is None and beta is None:
            print_callbacks[0]("Calculating distance matrix and scaling factors...")
            P, beta = UmapL1._x2p(X, perplexity=perplexity, print_callback=print_callbacks[1], n_threads=n_threads)
            print_callbacks[0]("Done.", tictoc.toc())
        elif P is None and beta is not None:
            print_callbacks[0]("Calculating distance matrix...")
            P = UmapL1._x2p_given_beta(X, beta)
            print_callbacks[0]("Done.", tictoc.toc())

        return P, beta

    def _fit_core(self, X, P, beta, must_keep, model_class: Type[_ABCTorchModel], tictoc):

        self.verbose_print(0, "Creating model without batches...")
        model = model_class(P, X, self.w, beta, self._torch_precision, self._torch_cdist_compute_mode,
                            self._t_distr, must_keep, ridge=self._ridge)
        if self._use_gpu:
            model.use_gpu()

        self.verbose_print(0, "Optimizing using LBFGS (because lasso is zero)...")
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe",
                                      max_iter=self._max_inner_iter, history_size=self._owlqn_history_size, lr=1.)

        if self._keep_fitting_info:
            self.model = model

        for t in range(self._max_outer_iter):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = model.forward()

                if loss.requires_grad:
                    loss.backward()

                return loss

            loss = optimizer.step(closure)
            self.verbose_print(1, t, 'loss (before this step):', loss.item(),
                               "Nonzero (after):", (np.abs(model.get_w0()) > self._eps).sum(),
                               tictoc.toc())

            self.w = model.get_w()

        loss = model.forward()
        self.verbose_print(1, 'Final', 'loss:', loss.item(), "Nonzero:", (np.abs(model.get_w0()) > self._eps).sum(),
                           tictoc.toc())

        # self.w = model.get_w()

        return self

    @staticmethod
    def _Hbeta(D=np.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """
        # Compute P-row and corresponding perplexity
        #P = np.zeros(D.shape)
        #k = 100
        #mask = np.argpartition(P, k)[:k]
        #P[mask] = np.exp(-(D[mask] - np.min(D[mask])) * beta)
        P = np.exp(-(D - np.min(D)) * beta)
        H = sum(P)
        return H, P

    @staticmethod
    def _x2p(X=np.array([]), tol=1e-5, perplexity=30.0, print_callback=print, *, n_threads):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """
        if n_threads > 1:
            return UmapL1._x2p_parallel(X, tol, perplexity, print_callback, n_threads)

        # Initialize some variables
        print_callback("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        D = np.sqrt(np.maximum(D, 0))
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            if i % 500 == 0:
                print_callback("Computing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = UmapL1._Hbeta(Di, beta[i])

            # if i % 500 == 0:
            # print(H, thisP)

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0

            while (not np.abs(Hdiff) < tol) and tries < 100:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] + 1.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = UmapL1._Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        print_callback("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P, beta

    @staticmethod
    def _x2p_given_beta(X, beta):
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        for i in range(n):
            (H, P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]) = UmapL1._Hbeta(
                D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], beta[i])
        return P

    @staticmethod
    def _x2p_process(Di, logU, tol):
        beta = 1.
        betamin = -np.inf
        betamax = np.inf
        (H, thisP) = UmapL1._Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0

        while (not np.abs(Hdiff) < tol) and tries < 100:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf or betamax == -np.inf:
                    beta = beta * 2.
                else:
                    beta = (beta + betamax) / 2.
            else:
                betamax = beta
                if betamin == np.inf or betamin == -np.inf:
                    beta = beta / 2.
                else:
                    beta = (beta + betamin) / 2.

            # Recompute the values
            (H, thisP) = UmapL1._Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1
        return thisP, beta
        # Set the final row of P

    @staticmethod
    def _x2p_parallel(X=np.array([]), tol=1e-5, perplexity=30.0, print_callback=print, n_threads=6):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        print_callback("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        D = np.sqrt(np.maximum(D, 0))

        # print(D)

        logU = np.log2(perplexity)

        # Loop over all datapoints
        # for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision

        print_callback("Using", n_threads, "threads...")
        parameters = [(D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], logU, tol) for i in range(n)]
        with multiprocessing.Pool(n_threads) as pool:
            results = pool.starmap(UmapL1._x2p_process, parameters)

        beta = np.ones((n, 1))
        P = np.zeros((n, n))
        for i in range(n):
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = results[i][0]
            beta[i] = results[i][1]

        # Return final P-matrix
        print_callback("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        # print(P)
        # print(beta)
        # raise NotImplementedError("...")
        return P, beta
