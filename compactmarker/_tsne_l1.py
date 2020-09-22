import torch
from typing import Type
from ._interfaces import _ABCSelector, _ABCTsneModel
from ._owlqn import OWLQN0

import warnings
import numpy as np

import scipy.stats
from sklearn.decomposition import PCA

from ._utils import TicToc, VerbosePrint
from ._torch_models import _RegTsneModel, _StratifiedRegTsneModel


class TsneL1(_ABCSelector):
    def __init__(self, w=None, lasso=1e-4, n_pcs=None, perplexity=30., use_beta_in_Q=True,
                 max_outer_iter=5, max_inner_iter=20, owlqn_history_size=100,
                 eps=1e-12, verbosity=2, torch_precision=32, torch_cdist_compute_mode="use_mm_for_euclid_dist",
                 t_distr=True):
        """

        :param w:
        :param lasso:
        :param n_pcs:
        :param perplexity:
        :param use_beta_in_Q:
        :param max_outer_iter:
        :param max_inner_iter:
        :param owlqn_history_size:
        :param eps:
        :param verbosity:
        :param torch_precision: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying. You can input 32, "32", 64, or "64".
        :param torch_cdist_compute_mode: cdist_compute_mode: compute mode for torch.cdist. By default,
            "use_mm_for_euclid_dist" to (daramatically) improve performance. However, if numerical stability became an
            issue, "donot_use_mm_for_euclid_dist" may be used instead. This option does not affect distances computed
            outside of pytorch, e.g., matrix P. Only matrix Q is affect.
        :param t_distr: By default, use t-distribution (1. / (1. + pdist2) for Q.
            Use Normal distribution instead (exp(-pdist2)) if set to False
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

    def fit(self, X, batches=None, use_beta_in_Q=False):
        """
        Select markers from one dataset to keep the cell-cell similarities in the same dataset
        :param X: data matrix (cells (rows) x genes/proteins (columns))
        :param batches: (optional) batch labels
        :param use_beta_in_Q:
        :return:
        """
        if batches is None:
            return self._fit(X)
        else:
            tictoc = TicToc()
            Xs, Ps, betas = self._resolve_batches(X, None, batches, self._n_pcs, self._perplexity, tictoc, self.verbose_print)
            return self._fit_core(Xs, Ps, betas, _StratifiedRegTsneModel, tictoc)

    def fit2(self, X_original, X_mock, use_beta_in_Q=False):
        """
        Select markers from one dataset to keep the cell-cell similarities in another dataset
        :param X_original: get target similarities from this dataset
        :param X_mock: choose markers from this dataset
        :param use_beta_in_Q:
        :return:
        """
        tictoc = TicToc()
        if self._n_pcs is None:
            P, beta = self.resolve_P_beta(X_original, None, None, self._perplexity, tictoc, self.verbose_print.prints)
        else:
            pcs = PCA(self._n_pcs).fit_transform(X_original)
            P, beta = self.resolve_P_beta(pcs, None, None, self._perplexity, tictoc, self.verbose_print.prints)
        return self._fit_core(X_mock, P, beta, _RegTsneModel, tictoc)

    def get_mask(self):
        return self.w > self._eps

    def transform(self, X):
        # if mask_only:
        return X[:, self.get_mask()]
        # else:
        #    return X[:, self.get_mask()] * self.w[self.get_mask()]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @staticmethod
    def resolve_P_beta(X, P, beta, perplexity, tictoc, print_callbacks):
        if P is None and beta is None:
            print_callbacks[0]("Calculating distance matrix and scaling factors...")
            P, beta = TsneL1.x2p(X, perplexity=perplexity, print_callback=print_callbacks[1])
            print_callbacks[0]("Done.", tictoc.toc())
        elif P is None and beta is not None:
            print_callbacks[0]("Calculating distance matrix...")
            P = TsneL1.x2p_given_beta(X, beta)
            print_callbacks[0]("Done.", tictoc.toc())

        return P, beta

    @classmethod
    def tune(cls, X, target_n_features, w=None, n_pcs=None, perplexity=30.,
             min_lasso=1e-8, max_lasso=1e-2,
             P=None, beta=None, torlerance=0, smallest_log10_fold_change=0.1, max_iter=100,
             max_outer_iter=5, max_inner_iter=20, owlqn_history_size=100, eps=1e-12, verbosity=2):
        """
        Automatically find proper lasso strength that returns the preferred number of markers
        :param X: Expression matrix, cells x features
        :param target_n_features: number of features
        :param w:
        :param min_lasso:
        :param max_lasso:
        :param P:
        :param beta:
        :param torlerance:
        :param smallest_log10_fold_change:
        :param max_iter:
        :param max_outer_iter:
        :param max_inner_iter:
        :param owlqn_history_size:
        :param eps:
        :param verbosity:
        :return: model
        """
        verbose_print = VerbosePrint(verbosity)
        tictoc = TicToc()

        n_features = X.shape[1]

        if w is None:
            w = np.random.uniform(size=[1, n_features])
        else:
            w = np.array(w).reshape([1, n_features])

        max_log_lasso = np.log10(max_lasso)
        min_log_lasso = np.log10(min_lasso)

        if n_pcs is None:
            P, beta = cls.resolve_P_beta(X, P, beta, perplexity, tictoc, verbose_print.prints)
        else:
            pcs = PCA(n_pcs).fit_transform(X)
            P, beta = cls.resolve_P_beta(pcs, None, None, perplexity, tictoc, verbose_print.prints)

        sup = n_features
        inf = 0

        for it in range(max_iter):
            log_lasso = max_log_lasso / 2 + min_log_lasso / 2
            verbose_print(0, "Iteration", it, "with lasso =", 10 ** log_lasso,
                          "in [", 10 ** min_log_lasso, ",", 10 ** max_log_lasso, "]...", end=" ")
            model = cls(w, 10 ** log_lasso, max_outer_iter, max_inner_iter, owlqn_history_size, eps, verbosity - 1)
            n = model._fit(X, P, beta).get_mask().sum()
            verbose_print(0, "Done. Number of features:", n, ".", tictoc.toc())
            if np.abs(n - target_n_features) <= torlerance:  # Good number of features, return
                break

            if it > 0 and np.abs(log_lasso - prev_log_lasso) < smallest_log10_fold_change:
                warnings.warn("smallest_log10_fold_change reached before achieving target number of features.")
                break

            prev_log_lasso = log_lasso

            if n > target_n_features:  # Too many features, need more l1 regularization
                if n <= sup:
                    sup = n
                else:
                    warnings.warn("Monotonicity is violated. Value larger than current supremum.")
                min_log_lasso = log_lasso
            elif n < target_n_features:  # Too few features, need less l1 regularization
                if n >= inf:
                    inf = n
                else:
                    warnings.warn("Monotonicity is violated. Value lower than current infimum.")
                max_log_lasso = log_lasso
        else:  # max_iter reached
            warnings.warn("max_iter before reached achieving target number of features.")

        return model

    def _fit(self, X, P=None, beta=None):
        tictoc = TicToc()
        if self._n_pcs is None:
            P, beta = self.resolve_P_beta(X, P, beta, self._perplexity, tictoc, self.verbose_print.prints)
        else:
            pcs = PCA(self._n_pcs).fit_transform(X)
            P, beta = self.resolve_P_beta(pcs, None, None, self._perplexity, tictoc, self.verbose_print.prints)

        self.verbose_print(0, "Optimizing...")
        return self._fit_core(X, P, beta, _RegTsneModel, tictoc)

    @staticmethod
    def _resolve_batches(X, beta, batches, n_pcs, perplexity, tictoc, verbose_print):
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
                P, new_beta = TsneL1.resolve_P_beta(Xs[-1], None, new_beta, perplexity, tictoc, verbose_print.prints)
            else:
                pcs = PCA(n_pcs).fit_transform(Xs[-1])
                P, new_beta = TsneL1.resolve_P_beta(pcs, None, None, perplexity, tictoc, verbose_print.prints)
            Ps.append(P)
            betas.append(new_beta)
        return Xs, Ps, betas

    def _fit_core(self, X, P, beta, model_class : Type[_ABCTsneModel], tictoc):

        self.verbose_print(0, "Optimizing...")
        if self._use_beta_in_Q:
            model = model_class(P, X, self.w, beta, self._torch_precision, self._torch_cdist_compute_mode, self._t_distr)
        else:
            model = model_class(P, X, self.w, beta, self._torch_precision, self._torch_cdist_compute_mode, self._t_distr)
        optimizer = OWLQN0(model.parameters(), lasso=self._lasso, line_search_fn="strong_wolfe",
                           max_iter=self._max_inner_iter, history_size=self._owlqn_history_size)
        self.model = model
        for t in range(self._max_outer_iter):
            def closure():
                # print(model.W)
                # print((np.abs(model.W.detach().numpy()) > self._eps).sum())
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = model.forward()
                # print(loss)
                # print(model.Q)
                if loss.requires_grad:
                    loss.backward()
                    # print(model.Q.grad)
                    # print((np.isnan(model.Q.grad.detach().numpy())).sum())
                    # print(model.W.grad)
                    # print((np.isnan(model.W.grad.detach().numpy())).sum())

                return loss

            loss = optimizer.step(closure)
            self.verbose_print(1, t, 'loss:', loss.item(), "Nonzero:", (np.abs(model.W.detach().numpy()) > self._eps).sum(),
                               tictoc.toc())

        loss = model.forward()
        self.verbose_print(1, 'final', 'loss:', loss.item(), "sparsity:", (np.abs(model.W.detach().numpy()) > self._eps).sum(),
                           tictoc.toc())

        self.w = model.W.detach().numpy().squeeze()

        return self

    @staticmethod
    def Hbeta(D=np.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        # print(sumP)
        # H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        H = scipy.stats.entropy(P)
        return H, P

    @staticmethod
    def x2p(X=np.array([]), tol=1e-5, perplexity=30.0, print_callback=print):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

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
            (H, thisP) = TsneL1.Hbeta(Di, beta[i])

            # if i % 500 == 0:
            # print(H, thisP)

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
                (H, thisP) = TsneL1.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        print_callback("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P, beta

    @staticmethod
    def x2p_given_beta(X, beta):
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        for i in range(n):
            (H, P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]) = TsneL1.Hbeta(
                D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], beta[i])
        return P
