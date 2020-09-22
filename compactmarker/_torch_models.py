from torch import nn
import torch
import numpy as np
import warnings

from ._interfaces import _ABCTsneModel


class _BaseTsneModel(nn.Module):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True):
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

        self.cdist_compute_mode = cdist_compute_mode
        self.t_distr = t_distr

    def init_w(self, w):
        if w is None:
            w = np.random.uniform(size=[1, self.n_features])
        else:
            w = np.array(w).reshape([1, self.n_features])
        self.W = torch.nn.Parameter(
            torch.tensor(w, dtype=self.dtype, requires_grad=True))


class _RegTsneModel(_BaseTsneModel, _ABCTsneModel):
    def __init__(self, P, X, w, beta, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True):
        super(_RegTsneModel, self).__init__(dtype, cdist_compute_mode, t_distr)
        self.n_instances, self.n_features = X.shape

        P = P + P.T
        P = P / np.sum(P)
        P = P * 4
        P = np.maximum(P, 1e-12)
        self.P = torch.tensor(P, dtype=self.dtype, requires_grad=False)
        self.X = torch.tensor(X, dtype=self.dtype, requires_grad=False)
        self.P /= 4
        if beta is not None:
            self.beta = torch.tensor(beta, dtype=self.dtype, requires_grad=False)
        else:
            self.beta = None

        self.init_w(w)

    def forward(self):
        P = self.P
        Y = self.X * self.W
        #Y.register_hook(lambda x: print("Y_grad", x, '\n', (np.isnan(x.numpy())).sum(),
        #                                (np.isinf(x.numpy())).sum(), (np.isinf(-x.numpy())).sum()))

        # Distance matrix calculation switched to torch.cdist for better numerical stability
        # TODO: if possible, modify torch.cdist to get a squared cdist. Have to deal with C so may not worth it
        # YY = torch.square(Y).sum(dim=1, keepdim=True)
        # print('YY shape', YY.shape)
        # pdist2 = YY - 2. * Y @ Y.T + YY.T
        pdist2 = torch.square(torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode))
        # pdist2.register_hook(lambda x: print("pdist2_grad", x, '\n', (np.isnan(x.numpy())).sum(),
        #                                (np.isinf(x.numpy())).sum(), (np.isinf(-x.numpy())).sum()))

        #print('min pdist2', pdist2.min())
        if self.beta is not None:
            pdist2 = pdist2 * self.beta
            temp = (pdist2 + pdist2.T) / 2.

        if self.t_distr:
            temp = 1. / (1. + pdist2)
        else:
            temp = torch.exp(-pdist2)

        temp[range(self.n_instances), range(self.n_instances)] = 0.
        #temp.register_hook(lambda x: print("temp_grad", x, '\n', (np.isnan(x.numpy())).sum(),
        #                                (np.isinf(x.numpy())).sum(), (np.isinf(-x.numpy())).sum()))

        Q = temp / temp.sum()
        Q = torch.max(Q, torch.tensor(1e-12 / 4, dtype=self.dtype))
        self.Q = Q
        #self.Q.register_hook(lambda x: print("Q_grad", x, '\n', (np.isnan(x.numpy())).sum(),
        #                                (np.isinf(x.numpy())).sum(), (np.isinf(-x.numpy())).sum()))
        kl = P * torch.log(P / Q)
        kl[range(self.n_instances), range(self.n_instances)] = 0.
        return kl.sum()


class _StratifiedRegTsneModel(_BaseTsneModel, _ABCTsneModel):

    def __init__(self, Ps, Xs, w, betas, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True):
        super(_StratifiedRegTsneModel, self).__init__(dtype, cdist_compute_mode, t_distr)

        self.n_batches = len(Xs)

        if not (len(Ps) == self.n_batches):
            raise ValueError("Lengths of Ps and Xs must be equal.")

        if betas is not None:
            if not (len(Ps) == self.n_batches):
                raise ValueError("Lengths of Xs and betas must be equal.")

        self.Ps = []
        for P in Ps:
            P = P + P.T
            P = P / np.sum(P)
            P = P * 4
            P = np.maximum(P, 1e-12)
            P /= 4
            self.Ps.append(torch.tensor(P, dtype=self.dtype, requires_grad=False))

        self.Xs = []
        self.n_instances = []
        self.n_features = Xs[0].shape[1]
        for X in Xs:
            self.Xs.append(torch.tensor(X, dtype=self.dtype, requires_grad=False))
            self.n_instances.append(X.shape[0])
            if X.shape[1] != self.n_features:
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

            Y = X * self.W
            # YY = (Y ** 2).sum(axis=1, keepdim=True)
            # pdist2 = YY - 2. * Y @ Y.T + YY.T
            pdist2 = torch.square(torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode))

            if self.beta is not None:
                pdist2 = pdist2 * self.beta
                temp = (pdist2 + pdist2.T) / 2.

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
