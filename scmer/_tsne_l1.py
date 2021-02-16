from typing import List, Union, Optional

import torch
from typing import Type
from ._interfaces import _ABCSelector, _ABCTorchModel
from ._owlqn import OWLQN

import warnings
import multiprocessing
import numpy as np

import scipy.stats
from sklearn.decomposition import PCA

from ._utils import TicToc, VerbosePrint
from ._tsne_torch_models import _RegTsneModel, _StratifiedRegTsneModel # , _SimpleRegTsneModel


class TsneL1(_ABCSelector):
    def __init__(self, *, w: Union[float, str, list, np.ndarray] = 'ones',
                 lasso: float = 1e-4, n_pcs: Optional[int] = None, perplexity: float = 30.,
                 use_beta_in_Q: bool = True,
                 max_outer_iter: int = 5, max_inner_iter: int = 20, owlqn_history_size: int = 100,
                 eps: float = 1e-12, verbosity: int = 2, torch_precision: Union[int, str, torch.dtype] = 32,
                 torch_cdist_compute_mode: str = "use_mm_for_euclid_dist",
                 t_distr: bool = True, n_threads: int = 1, use_gpu: bool = False, pca_seed=0, ridge=0.):
        """
        TsneL1 model

        :param w: initial value of w, weight of each marker. Acceptable values are 'ones' (all 1),
            'uniform' (random [0, 1] values), float numbers (all set to that number),
            or a list or numpy array with specific numbers.
        :param lasso: lasso strength
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
        :param t_distr: By default, use t-distribution (1. / (1. + pdist2) for Q.
            Use Normal distribution instead (exp(-pdist2)) if set to False
        :param n_threads: number of threads (currently only for calculating P and beta)
        :param use_gpu: whether to use GPU to train the model.
        """
        super(TsneL1, self).__init__(verbosity)
        self._max_outer_iter = max_outer_iter
        self._max_inner_iter = max_inner_iter
        self._owlqn_history_size = owlqn_history_size
        self._n_pcs = n_pcs
        self.w = w
        self._lasso = lasso
        self._eps = eps
        self._use_beta_in_Q = use_beta_in_Q
        self._perplexity = perplexity
        self._torch_precision = torch_precision
        self._torch_cdist_compute_mode = torch_cdist_compute_mode
        self._t_distr = t_distr
        self._n_threads = n_threads
        self._use_gpu = use_gpu
        self._pca_seed = pca_seed
        self._ridge = ridge

    def fit(self, X, *, X_teacher=None, batches=None, P=None, beta=None, must_keep=None):
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

        if X_teacher is None: # if there is no other assay to mimic, just mimic itself
            X_teacher = X

        if batches is None:
            if must_keep is None and (isinstance(self._lasso, float) or isinstance(self._lasso, str)):
                model_class = _RegTsneModel  # _SimpleRegTsneModel
            else:
                model_class = _RegTsneModel
            if self._n_pcs is None:
                P, beta = self._resolve_P_beta(X_teacher, P, beta, self._perplexity, tictoc, self.verbose_print.prints,
                                               self._n_threads)
            else:
                pcs = PCA(self._n_pcs, random_state=self._pca_seed).fit_transform(X_teacher)
                P, beta = self._resolve_P_beta(pcs, P, beta, self._perplexity, tictoc, self.verbose_print.prints,
                                               self._n_threads)
        else:
            model_class = _StratifiedRegTsneModel
            if P is None:
                X, P, beta = self._resolve_batches(X_teacher, None, batches, self._n_pcs, self._perplexity, tictoc,
                                                   self.verbose_print, self._n_threads)

        return self._fit_core(X, P, beta, must_keep, model_class, tictoc)

    @classmethod
    def tune(cls, target_n_features, X=None, *, X_teacher=None, batches=None,
             P=None, beta=None, must_keep=None, perplexity=30., n_pcs=None, w='ones',
             min_lasso=1e-8, max_lasso=1e-2, tolerance=0, smallest_log10_fold_change=0.1, max_iter=100,
             return_P_beta=False, n_threads=6,
             **kwargs):
        """
        Automatically find proper lasso strength that returns the preferred number of markers

        :param target_n_features: number of features
        :param return_P_beta: controls what to return
        :param kwargs: all other parameters are the same for a TsneL1 model or :func:`TsneL1.fit`.
        :return: if return_P_beta is True and there are batches, (model, X, P, beta);
                 if return_P_beta is True and there is no batches, (model, P, beta);
                 otherwise, only model by default.
        """
        if "lasso" in kwargs:
            raise ValueError("Parameter lasso should be substituted by max_lasso and min_lasso to set a range.")
        if "verbosity" in kwargs:
            verbosity = kwargs['verbosity']
        else:
            verbosity = 3
        verbose_print = VerbosePrint(verbosity)
        tictoc = TicToc()

        n_features = X.shape[1]

        # initialize w
        if isinstance(w, float) or isinstance(w, int):
            w = np.zeros([1, n_features]) + w
        elif isinstance(w, str) and w == 'uniform':
            w = np.random.uniform(size=[1, n_features])
        elif isinstance(w, str) and w == 'ones':
            w = np.ones([1, n_features])
        else:
            w = np.array(w).reshape([1, n_features])

        max_log_lasso = np.log10(max_lasso)
        min_log_lasso = np.log10(min_lasso)

        if X_teacher is None: # if there is no other assay to mimic, just mimic itself
            X_teacher = X

        if batches is None:
            model_class = _RegTsneModel
            if n_pcs is None:
                P, beta = cls._resolve_P_beta(X_teacher, P, beta, perplexity, tictoc, verbose_print.prints, n_threads)
            else:
                pcs = PCA(n_pcs).fit_transform(X_teacher)
                P, beta = cls._resolve_P_beta(pcs, P, beta, perplexity, tictoc, verbose_print.prints, n_threads)
        else:
            model_class = _StratifiedRegTsneModel
            if P is None:
                X, P, beta = cls._resolve_batches(X_teacher, None, batches, n_pcs, perplexity, tictoc, verbose_print,
                                                  n_threads)

        sup = n_features
        inf = 0

        model = None
        for it in range(max_iter):
            log_lasso = max_log_lasso / 2 + min_log_lasso / 2
            verbose_print(0, "Iteration", it, "with lasso =", 10 ** log_lasso,
                          "in [", 10 ** min_log_lasso, ",", 10 ** max_log_lasso, "]...", end=" ")
            model = cls(w=w, lasso=10 ** log_lasso, n_pcs=n_pcs, perplexity=perplexity, **kwargs)
            n = model._fit_core(X, P, beta, must_keep, model_class, tictoc).get_mask().sum()
            verbose_print(0, "Done. Number of features:", n, ".", tictoc.toc())
            if np.abs(n - target_n_features) <= tolerance:  # Good number of features, return
                break

            if it > 0 and np.abs(log_lasso - prev_log_lasso) < smallest_log10_fold_change:
                warnings.warn("smallest_log10_fold_change reached before achieving target number of features.")
                break

            prev_log_lasso = log_lasso

            if n > target_n_features:  # Too many features, need more l1 regularization
                if n <= sup:
                    sup = n
                else:
                    warnings.warn("Monotonicity is violated. Value larger than current supremum. "
                                  "Binary search may fail. "
                                  "Consider use more max_outer_iter (default: 5) and max_inner_iter (default: 20).")
                min_log_lasso = log_lasso
            elif n < target_n_features:  # Too few features, need less l1 regularization
                if n >= inf:
                    inf = n
                else:
                    warnings.warn("Monotonicity is violated. Value lower than current infimum. "
                                  "Binary search may fail. "
                                  "Consider use more max_outer_iter (default: 5) and max_inner_iter (default: 20).")
                max_log_lasso = log_lasso
        else:  # max_iter reached
            warnings.warn("max_iter before reached achieving target number of features.")

        if return_P_beta:
            if batches is None:
                return model, P, beta
            else:
                return model, X, P, beta
        else:
            return model

    def get_mask(self):
        """
        Get the feature selection mask.
        For AnnData in scanpy, it can be used as adata[:, model.get_mask()]

        :return: mask
        """
        return self.w > self._eps

    def transform(self, X):
        """
        Shrink a matrix / AnnData object with full markers to the selected markers only.
        If such operation is not supported by your data object,
        you can do it manually using :func:`~TsneL1.get_mask`.

        :param X: Matrix / AnnData to be shrunk
        :return: Shrunk matrix / Anndata
        """
        # if mask_only:
        return X[:, self.get_mask()]
        # else:
        #    return X[:, self.get_mask()] * self.w[self.get_mask()]

    def fit_transform(self, X, **kwargs):
        """
        Fit on a matrix / AnnData and then transfer it.

        :param X: The matrix / AnnData to be transformed
        :param kwargs: Other parameters for :func:`TsneL1.fit`.
        :return: Shrunk matrix / Anndata
        """
        return self.fit(X, **kwargs).transform(X)

    @staticmethod
    def _resolve_P_beta(X, P, beta, perplexity, tictoc, print_callbacks, n_threads):
        if P is None and beta is None:
            print_callbacks[0]("Calculating distance matrix and scaling factors...")
            P, beta = TsneL1._x2p(X, perplexity=perplexity, print_callback=print_callbacks[1], n_threads=n_threads)
            print_callbacks[0]("Done.", tictoc.toc())
        elif P is None and beta is not None:
            print_callbacks[0]("Calculating distance matrix...")
            P = TsneL1._x2p_given_beta(X, beta)
            print_callbacks[0]("Done.", tictoc.toc())

        return P, beta

    @staticmethod
    def _resolve_batches(X, beta, batches, n_pcs, perplexity, tictoc, verbose_print, n_threads):
        batches = np.array(batches)
        batch_names = np.unique(batches)
        Xs = []
        Ps = []
        betas = []
        for batch in batch_names:
            batch_mask = (batches == batch)
            verbose_print(0, "Batch", batch, "with", sum(batch_mask), "instances.")

            Xs.append(X[batch_mask, :])
            if n_pcs is None:
                if beta is not None:
                    new_beta = beta[batches == batch]
                else:
                    new_beta = None
                P, new_beta = TsneL1._resolve_P_beta(Xs[-1], None, new_beta, perplexity, tictoc, verbose_print.prints, n_threads)
            else:
                pcs = PCA(n_pcs).fit_transform(Xs[-1])
                P, new_beta = TsneL1._resolve_P_beta(pcs, None, None, perplexity, tictoc, verbose_print.prints, n_threads)
            Ps.append(P)
            betas.append(new_beta)
        return Xs, Ps, betas

    def _fit_core(self, X, P, beta, must_keep, model_class: Type[_ABCTorchModel], tictoc):

        self.verbose_print(0, "Optimizing...")
        if self._use_beta_in_Q:
            model = model_class(P, X, self.w, beta, self._torch_precision, self._torch_cdist_compute_mode,
                                self._t_distr, must_keep, self._ridge)
        else:
            model = model_class(P, X, self.w, None, self._torch_precision, self._torch_cdist_compute_mode,
                                self._t_distr, must_keep, self._ridge)

        if self._use_gpu:
            model.use_gpu()

        optimizer = OWLQN(model.parameters(), lasso=self._lasso, line_search_fn="strong_wolfe",
                          max_iter=self._max_inner_iter, history_size=self._owlqn_history_size)

        for t in range(self._max_outer_iter):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = model.forward()
                if loss.requires_grad:
                    loss.backward()

                return loss

            loss = optimizer.step(closure)
            self.verbose_print(1, t, 'loss:', loss.item(), "Nonzero:", (np.abs(model.get_w0()) > self._eps).sum(),
                               tictoc.toc())
            self.w = model.get_w()

        loss = model.forward()
        self.verbose_print(1, 'final', 'loss:', loss.item(), "Nonzero:", (np.abs(model.get_w0()) > self._eps).sum(),
                           tictoc.toc())

        return self

    @staticmethod
    def _Hbeta(D=np.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        P = P / sumP
        H = scipy.stats.entropy(P)
        return H, P

    @staticmethod
    def _x2p(X=np.array([]), tol=1e-5, perplexity=30.0, print_callback=print, *, n_threads):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """
        if n_threads > 1:
            return TsneL1._x2p_parallel(X, tol, perplexity, print_callback, n_threads)

        # Initialize some variables
        print_callback("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
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
            (H, thisP) = TsneL1._Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0

            while (not np.abs(Hdiff) < tol) and tries < 50:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = TsneL1._Hbeta(Di, beta[i])
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
            (H, P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]) = TsneL1._Hbeta(
                D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], beta[i])
        return P

    @staticmethod
    def _x2p_process(Di, logU, tol):
        beta = 1.
        betamin = -np.inf
        betamax = np.inf
        (H, thisP) = TsneL1._Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0

        while (not np.abs(Hdiff) < tol) and tries < 50:
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
            (H, thisP) = TsneL1._Hbeta(Di, beta)
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

        logU = np.log(perplexity)

        # Loop over all datapoints
        # for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision

        print_callback("Using", n_threads, "threads...")
        parameters = [(D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], logU, tol) for i in range(n)]
        with multiprocessing.Pool(n_threads) as pool:
            results = pool.starmap(TsneL1._x2p_process, parameters)

        beta = np.ones((n, 1))
        P = np.zeros((n, n))
        for i in range(n):
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = results[i][0]
            beta[i] = results[i][1]

        # Return final P-matrix
        print_callback("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P, beta
