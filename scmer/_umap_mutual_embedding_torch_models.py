from torch import nn
import torch
import numpy as np
import warnings


class _BaseMutualEmbeddingModel(nn.Module):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, ridge=0.):
        """
        Base class for umap models
        :param cdist_compute_mode: compute mode for torch.cdist. By default, "use_mm_for_euclid_dist" to (daramatically)
            improve performance. However, if numerical stability became an issue, "donot_use_mm_for_euclid_dist" may be
            used.
        :param dtype: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying.
        """
        super(_BaseMutualEmbeddingModel, self).__init__()
        if dtype == "32" or dtype == 32:
            self.dtype = torch.float32
        elif dtype == "64" or dtype == 64:
            self.dtype = torch.float64
        else:
            self.dtype = dtype

        self.epsilon = torch.tensor(1e-30, dtype=self.dtype)
        self.cdist_compute_mode = cdist_compute_mode
        self.t_distr = t_distr
        self.ridge = ridge

    @staticmethod
    def preprocess_P(P):
        P = P + P.T - P * P.T
        P = P / np.sum(P)
        P = np.maximum(P, 0.)
        #P = np.maximum(P, 1e-12)
        return P

    def calc_kl(self, P, Y):
        pdist2 = torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)

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

    def init_Y(self, Y, n_dims, n_instances):
        if isinstance(Y, str) and Y == 'normal':
            Y = np.random.normal(size=[n_instances, n_dims])
        else:
            Y = np.array(Y)
        return torch.nn.Parameter(
            torch.tensor(Y, dtype=self.dtype, requires_grad=True))

    def preprocess_X(self, X):
        X = torch.tensor(X, dtype=self.dtype, requires_grad=False)
        n_instances, n_features = X.shape
        return X, n_instances, n_features


class _MutualEmbeddingModel(_BaseMutualEmbeddingModel):
    def __init__(self, P, X, Y, beta, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist",
                 t_distr=True, n_dims=2, ridge=0.):
        super(_MutualEmbeddingModel, self).__init__(dtype, cdist_compute_mode, t_distr, ridge)

        self.P = torch.tensor(self.preprocess_P(P), dtype=self.dtype, requires_grad=False)
        self.X, self.n_instances, self.n_features = self.preprocess_X(X)
        self.Y = self.init_Y(Y, n_dims, self.n_instances)

        if beta is not None:
            self.beta = torch.tensor(beta, dtype=self.dtype, requires_grad=False)
        else:
            self.beta = None

    def forward(self):
        kl = self.calc_kl(self.P, self.Y)
        if self.ridge > 0.:
            return kl + torch.sum(self.Y ** 2) * self.ridge
        else:
            return kl

    def use_gpu(self):
        self.P = self.P.cuda()
        self.X = self.X.cuda()
        self.epsilon = self.epsilon.cuda()
        self.cuda()
