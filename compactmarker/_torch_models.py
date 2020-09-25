from torch import nn
import torch
import numpy as np
import warnings

from ._interfaces import _ABCTsneModel


class _BaseTsneModel(nn.Module):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None):
        """
        Base class for tsne models
        :param cdist_compute_mode: compute mode for torch.cdist. By default, "use_mm_for_euclid_dist" to (daramatically)
            improve performance. However, if numerical stability became an issue, "donot_use_mm_for_euclid_dist" may be
            used.
        :param dtype: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying.
        """
        super(_BaseTsneModel, self).__init__()
        if dtype == "32" or dtype == 32:
            self.dtype = torch.float32
        elif dtype == "64" or dtype == 64:
            self.dtype = torch.float64
        else:
            self.dtype = dtype

        if must_keep is None:
            self.must_keep = None
        else:
            self.must_keep = must_keep.squeeze()
        self.cdist_compute_mode = cdist_compute_mode
        self.t_distr = t_distr

    @staticmethod
    def preprocess_P(P):
        P = P + P.T
        P = P / np.sum(P)
        P = P * 4
        P = np.maximum(P, 1e-12)
        P /= 4
        return P

    def preprocess_X(self, X):
        if self.must_keep is None:
            add_pdist2 = 0.
            X = torch.tensor(X, dtype=self.dtype, requires_grad=False)
            n_instances, n_features = X.shape
        else:
            must_keep = self.must_keep
            add_X = torch.tensor(X * must_keep.reshape([1, -1]), dtype=self.dtype, requires_grad=False)
            add_pdist2 = torch.square(torch.cdist(add_X, add_X, compute_mode=self.cdist_compute_mode))
            X = torch.tensor(X[:, must_keep == 0], dtype=self.dtype, requires_grad=False)
            n_instances, n_features = X.shape
            n_features = (must_keep == 0).sum()
        return X, add_pdist2, n_instances, n_features

    def init_w(self, w):
        if w == 'uniform':
            w = np.random.uniform(size=[1, self.n_features])
        elif w == 'ones':
            w = np.ones([1, self.n_features])
        else:
            w = np.array(w).reshape([1, self.n_features])
        self.W = torch.nn.Parameter(
            torch.tensor(w, dtype=self.dtype, requires_grad=True))

    def get_w(self):
        if self.must_keep is None:
            return self.W.detach().numpy().squeeze()
        else:
            w = self.must_keep.copy()
            w[self.must_keep == 0] += self.W.detach().numpy().squeeze()
        return w

class _RegTsneModel(_BaseTsneModel, _ABCTsneModel):
    def __init__(self, P, X, w, beta, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None):
        super(_RegTsneModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep)

        self.P = torch.tensor(self.preprocess_P(P), dtype=self.dtype, requires_grad=False)
        self.X, self.add_pdist2, self.n_instances, self.n_features = self.preprocess_X(X)

        if beta is not None:
            self.beta = torch.tensor(beta, dtype=self.dtype, requires_grad=False)
        else:
            self.beta = None

        self.init_w(w)

    def forward(self):
        P = self.P
        Y = self.X * self.W

        pdist2 = torch.square(torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)) + self.add_pdist2
        if self.beta is not None:
            pdist2 = pdist2 * self.beta
            pdist2 = (pdist2 + pdist2.T) / 2.

        if self.t_distr:
            temp = 1. / (1. + pdist2)
        else:
            temp = torch.exp(-pdist2)

        temp[range(self.n_instances), range(self.n_instances)] = 0.

        Q = temp / temp.sum()
        Q = torch.max(Q, torch.tensor(1e-12 / 4, dtype=self.dtype))
        self.Q = Q
        kl = P * torch.log(P / Q)
        kl[range(self.n_instances), range(self.n_instances)] = 0.
        return kl.sum()


class _StratifiedRegTsneModel(_BaseTsneModel, _ABCTsneModel):
    def __init__(self, Ps, Xs, w, betas, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None):
        super(_StratifiedRegTsneModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep)

        self.n_batches = len(Xs)

        if not (len(Ps) == self.n_batches):
            raise ValueError("Lengths of Ps and Xs must be equal.")

        if betas is not None:
            if not (len(Ps) == self.n_batches):
                raise ValueError("Lengths of Xs and betas must be equal.")

        self.Ps = [torch.tensor(self.preprocess_P(P), dtype=self.dtype, requires_grad=False) for P in Ps]

        self.Xs = []
        self.add_pdist2s = []
        self.n_instances = []
        self.n_features = None
        for X in Xs:
            X, add_pdist2, n_instances, n_features = self.preprocess_X(X)
            self.Xs.append(X)
            self.add_pdist2s.append(add_pdist2)
            self.n_instances.append(n_instances)
            if self.n_features is None:
                self.n_features = n_features
            elif self.n_features != n_features:
                raise ValueError("All matrices must have the same number of features.")

        self.betas = []
        if betas is not None:
            for beta in betas:
                self.betas.append(torch.tensor(beta, dtype=self.dtype, requires_grad=False))
        else:
            self.betas = None

        self.init_w(w)

    def forward(self):
        loss = 0
        for batch in range(self.n_batches):
            P = self.Ps[batch]
            X = self.Xs[batch]
            n_instances = self.n_instances[batch]
            add_pdist2 = self.add_pdist2s[batch]

            Y = X * self.W
            pdist2 = torch.square(torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)) + add_pdist2

            if self.beta is not None:
                pdist2 = pdist2 * self.beta
                pdist2 = (pdist2 + pdist2.T) / 2.

            if self.t_distr:
                temp = 1. / (1. + pdist2)
                temp[range(n_instances), range(n_instances)] = 0.
            else:
                temp = torch.exp(-pdist2)
                temp[range(n_instances), range(n_instances)] = 0.

            Q = temp / temp.sum()
            Q = torch.max(Q, torch.tensor(1e-12, dtype=self.dtype))
            kl = P * torch.log(P / Q)
            kl[range(n_instances), range(n_instances)] = 0.
            loss += kl.sum()
        return loss / self.n_batches
