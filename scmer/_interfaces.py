from abc import ABC, abstractmethod
from ._utils import VerbosePrint
from torch import nn


class _ABCSelector(ABC):
    @abstractmethod
    def __init__(self, verbosity):
        self.verbose_print = VerbosePrint(verbosity)

    @abstractmethod
    def fit(self, X, **kwargs):
        pass

    @abstractmethod
    def transform(self, X, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, X, **kwargs):
        pass


class _ABCTorchModel(ABC):
    @abstractmethod
    def __init__(self, P, X, w, beta, dtype, cdist_compute_mode, t_distr, must_keep, ridge):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def get_w(self):
        pass
    
    @abstractmethod
    def use_gpu(self):
        pass
