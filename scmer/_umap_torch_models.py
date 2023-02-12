from torch import nn
import torch
import numpy as np
import warnings

from ._base_torch_model import _BaseTorchModel


class _BaseUmapModel(_BaseTorchModel):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None, ridge=0.):
        """
        Base class for umap models
        :param cdist_compute_mode: compute mode for torch.cdist. By default, "use_mm_for_euclid_dist" to (daramatically)
            improve performance. However, if numerical stability became an issue, "donot_use_mm_for_euclid_dist" may be
            used.
        :param dtype: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying.
        """
        super(_BaseUmapModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep, ridge)

        self.epsilon = torch.tensor(1e-30, dtype=self.dtype)

    @staticmethod
    def preprocess_P(P):
        P = P + P.T - P * P.T
        P = P / np.sum(P)
        P = np.maximum(P, 0.)

        return P

    def calc_kl(self, P, X, W, beta, add_pdist2):
        Y = X * W
        if isinstance(add_pdist2, float):
            pdist2 = torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)
        else:
            pdist2 = torch.sqrt(torch.square(torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)) + add_pdist2)
        pdist2 = pdist2 - torch.min(pdist2.clone().fill_diagonal_(float("inf")), dim=1, keepdim=True)[0]

        if beta is not None:
            pdist2 = pdist2 * beta

        temp = 1. / (1. + pdist2)
        temp = temp + temp.T - temp * temp.T
        temp.fill_diagonal_(0.)

        Q = temp / temp.sum()
        Q = torch.max(Q, self.epsilon)

        mask = P > 0.

        Q = Q[mask]
        P = P[mask]

        kl = P * torch.log(P / Q)
        kl = kl.sum()
        return kl.sum()


class _RegUmapModel(_BaseUmapModel):
    def __init__(self, P, X, w, beta, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist",
                 t_distr=True, must_keep=None, ridge=0.):
        super(_RegUmapModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep, ridge)

        self.P = torch.tensor(self.preprocess_P(P), dtype=self.dtype, requires_grad=False)
        self.X, self.add_pdist2, self.n_instances, self.n_features = self.preprocess_X(X)

        if beta is not None:
            self.beta = torch.tensor(beta, dtype=self.dtype, requires_grad=False)
        else:
            self.beta = None

        self.init_w(w)

    def forward(self):
        kl = self.calc_kl(self.P, self.X, self.W, self.beta, self.add_pdist2)
        if self.ridge > 0.:
            return kl + torch.sum(self.W ** 2) * self.ridge
        else:
            return kl

    def use_gpu(self):
        self.P = self.P.cuda()
        self.X = self.X.cuda()
        self.epsilon = self.epsilon.cuda()
        if self.beta is not None:
            self.beta = self.beta.cuda()
        if not isinstance(self.add_pdist2, float):
            self.add_pdist2 = self.add_pdist2.cuda()
        self.cuda()


class _StratifiedRegUmapModel(_BaseUmapModel):
    def __init__(self, Ps, Xs, w, betas, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist",
                 t_distr=True, must_keep=None, ridge=0.):
        super(_StratifiedRegUmapModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep, ridge)

        self.n_batches = len(Xs)

        if not (len(Ps) == self.n_batches):
            raise ValueError("Lengths of Ps and Xs must be equal.")

        if betas is not None:
            self.betas = betas
            if not (len(Ps) == self.n_batches):
                raise ValueError("Lengths of Xs and betas must be equal.")
        else:
            self.betas = betas

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
            loss += self.calc_kl(self.Ps[batch], self.Xs[batch], self.W,
                                 self.betas[batch] if self.betas is not None else None,
                                 self.add_pdist2s[batch])
        if self.ridge > 0.:
            return loss / self.n_batches + torch.sum(self.W ** 2) * self.ridge
        else:
            return loss / self.n_batches

    def use_gpu(self):
        self.Ps = [P.cuda() for P in self.Ps]
        self.Xs = [X.cuda() for X in self.Xs]
        self.epsilon = self.epsilon.cuda()
        if self.betas is not None:
            self.betas = [beta.cuda() for beta in self.betas]
        self.add_pdist2s = [add_pdist2 if isinstance(add_pdist2, float) else add_pdist2.cuda()
                            for add_pdist2 in self.add_pdist2s]
        self.cuda()
