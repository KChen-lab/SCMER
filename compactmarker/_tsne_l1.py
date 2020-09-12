from ._base import _BaseSelector
from ._owlqn import OWLQN0

import warnings
import numpy as np
from torch import nn
import torch
import scipy.stats

from ._utils import TicToc, VerbosePrint


class _RegTsneModel(nn.Module):

    def __init__(self, P, X, w):
        super(_RegTsneModel, self).__init__()

        if w is None:
            w = np.random.uniform(size=[1, self.n_features])

        P = P + P.T
        P = P / np.sum(P)
        P = P * 4
        P = np.maximum(P, 1e-12)
        self.P = torch.tensor(P, dtype=torch.float, requires_grad=False)
        self.X = torch.tensor(X, dtype=torch.float, requires_grad=False)
        self.P /= 4
        self.n_instances, self.n_features = X.shape
        self.W = torch.nn.Parameter(
            torch.tensor(w, dtype=torch.float, requires_grad=True))

    def forward(self):
        P = self.P
        Y = self.X * self.W
        YY = (Y ** 2).sum(axis=1, keepdim=True)
        pdist2 = YY - 2. * Y @ Y.T + YY.T
        temp = 1. / (1. + pdist2)
        temp[range(self.n_instances), range(self.n_instances)] = 0.
        Q = temp / temp.sum()
        Q = torch.max(Q, torch.tensor(1e-12, dtype=torch.float))
        self.Q = Q
        return (P * torch.log(P / Q)).sum()


class TsneL1(_BaseSelector):
    def __init__(self, w=None, lasso=1e-4, max_outer_iter=5, max_inner_iter=20, owlqn_history_size=100, eps=1e-12,
                 verbosity=2):
        super(TsneL1, self).__init__(verbosity)
        self._max_outer_iter = max_outer_iter
        self._max_inner_iter = max_inner_iter
        self._owlqn_history_size = owlqn_history_size
        self.w = w
        self._lasso = lasso
        self._eps = eps

    def fit(self, X, w=None):
        return self._fit(X, w)

    def get_mask(self):
        return self.w > self._eps

    def transform(self, X):
        return X[:, self.get_mask()]

    def fit_transform(self, X, w=None):
        return self.fit(X, w).transform(X)

    @staticmethod
    def resolve_P_beta(X, P, beta, tictoc, print_callbacks):
        if P is None and beta is None:
            print_callbacks[0]("Calculating distance matrix and scaling factors...")
            P, beta = TsneL1.x2p(X, print_callback=print_callbacks[1])
            print_callbacks[0]("Done.", tictoc.toc())
        elif P is None and beta is not None:
            print_callbacks[0]("Calculating distance matrix...")
            P = TsneL1.x2p_given_beta(X, beta)
            print_callbacks[0]("Done.", tictoc.toc())

        return P, beta

    @classmethod
    def tune(cls, X, n_features, w=None,
             init_lasso=1e-5, min_lasso=1e-8, max_lasso=1e-2,
             P=None, beta=None, torlerance=0, smallest_log10_fold_change=0.1, max_iter=100,
             max_outer_iter=5, max_inner_iter=20, owlqn_history_size=100, eps=1e-12, verbosity=2):
        """

        :param X:
        :param n_features:
        :param w:
        :param init_lasso:
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

        log_lasso = np.log10(init_lasso)
        max_log_lasso = np.log10(max_lasso)
        min_log_lasso = np.log10(min_lasso)

        P, beta = cls.resolve_P_beta(X, P, beta, tictoc, verbose_print.prints)

        for it in range(max_iter):
            verbose_print(0, "Iteration", it, "with lasso =", 10 ** log_lasso, "...", end=" ")
            model = cls(w, 10 ** log_lasso, max_outer_iter, max_inner_iter, owlqn_history_size, eps, verbosity - 1)
            n = model._fit(X, w, P, beta).get_mask().sum()
            verbose_print(0, "Done. Number of features:", n, ".")
            if np.abs(n - n_features) <= torlerance:  # Good number of features, return
                break

            if it > 0 and np.abs(log_lasso - prev_log_lasso) < smallest_log10_fold_change:
                warnings.warn("smallest_log10_fold_change reached before achieving target number of features.")
                break

            prev_log_lasso = log_lasso

            if n > n_features and n - n_features > torlerance:  # Too many features, need more l1 regularization
                log_lasso = max_log_lasso / 2 + log_lasso / 2
            elif n < n_features and n_features - n < torlerance:  # Too few features, need less l1 regularization
                log_lasso = min_log_lasso / 2 + log_lasso / 2
        else:  # max_iter reached
            warnings.warn("max_iter before reached achieving target number of features.")

        return model

    def _fit(self, X, w, P=None, beta=None):
        """

        :param X:
        :param w:
        :return:
        """

        tictoc = TicToc()
        P, beta = self.resolve_P_beta(X, P, beta, tictoc, self.verbose_print.prints)

        self.verbose_print(0, "Optimizing...")
        model = _RegTsneModel(P, X, self.w)
        optimizer = OWLQN0(model.parameters(), lasso=self._lasso, line_search_fn="strong_wolfe",
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
            self.verbose_print(1, t, 'loss:', loss, "Nonzero:", (np.abs(model.W.detach().numpy()) > self._eps).sum(),
                               tictoc.toc())

        loss = model.forward()
        self.verbose_print(1, 'final', 'loss:', loss, "sparsity:", (np.abs(model.W.detach().numpy()) > self._eps).sum(),
                           tictoc.toc())

        self.w = model.W.detach().numpy()

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
