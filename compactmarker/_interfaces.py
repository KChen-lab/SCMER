from abc import ABC, abstractmethod
from ._utils import VerbosePrint


class _ABCSelector(ABC):
    @abstractmethod
    def __init__(self, verbosity):
        self.verbose_print = VerbosePrint(verbosity)

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass


class _ABCTsneModel(ABC):
    @abstractmethod
    def __init__(self, P, X, w, beta, dtype, cdist_compute_mode, t_distr, must_keep):
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