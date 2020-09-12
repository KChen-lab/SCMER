from abc import ABC, abstractmethod, abstractproperty
from ._utils import VerbosePrint

class _BaseSelector(ABC):
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
