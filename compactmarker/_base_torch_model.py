import numpy as np
import torch
from torch import nn

from ._interfaces import _ABCTorchModel


class _BaseTorchModel(nn.Module, _ABCTorchModel):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None, ridge=0.):
        super(_BaseTorchModel, self).__init__()
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
        self.ridge = ridge

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
        if isinstance(w, float) or isinstance(w, int):
            w = np.zeros([1, self.n_features]) + w
        elif isinstance(w, str) and w == 'uniform':
            w = np.random.uniform(size=[1, self.n_features])
        elif isinstance(w, str) and w == 'ones':
            w = np.ones([1, self.n_features])
        else:
            w = np.array(w).reshape([1, self.n_features])
        self.W = torch.nn.Parameter(
            torch.tensor(w, dtype=self.dtype, requires_grad=True))

    def get_w0(self):
        if self.W.is_cuda:
            w0 = self.W.detach().cpu().numpy().squeeze()
        else:
            w0 = self.W.detach().numpy().squeeze()
        return w0

    def get_w(self):
        w0 = self.get_w0()
        if self.must_keep is None:
            w = w0
        else:
            w = self.must_keep.copy()
            w[self.must_keep == 0] += w0
        return w