from ._interfaces import _ABCSelector
import torch
from typing import Union, Optional
import numpy as np


class _BaseSelector(_ABCSelector):
    def __init__(self, w: Union[float, str, list, np.ndarray] = 'ones',
                 lasso: float = 1e-4, n_pcs: Optional[int] = None, perplexity: float = 30.,
                 use_beta_in_Q: bool = True,
                 max_outer_iter: int = 5, max_inner_iter: int = 20, owlqn_history_size: int = 100,
                 eps: float = 1e-12, verbosity: int = 2, torch_precision: Union[int, str, torch.dtype] = 32,
                 torch_cdist_compute_mode: str = "use_mm_for_euclid_dist",
                 t_distr: bool = True, n_threads: int = 1, use_gpu: bool = False, pca_seed: int = 0, ridge: float = 0.,
                 _keep_fitting_info: bool = False):
        super(_BaseSelector, self).__init__(verbosity)
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
        self._n_threads = n_threads
        self._use_gpu = use_gpu
        self._pca_seed = pca_seed
        self._ridge = ridge
        self._keep_fitting_info = _keep_fitting_info

    def get_mask(self, target_n_features=None):
        """
        Get the feature selection mask.
        For AnnData in scanpy, it can be used as adata[:, model.get_mask()]

        :param target_n_features: If None, all features with w > 0 are selected. If not None, only select
            `target_n_features` largest features
        :return: mask
        """
        if target_n_features is None:
            return self.w > 0.
        else:
            n_nonzero = (self.w > 0.).sum()
            if target_n_features > n_nonzero:
                raise ValueError(f"Only {n_nonzero} features have nonzero weights. "
                                 f"target_n_features may not exceed the number.")
            return self.w >= self.w[np.argpartition(self.w, -target_n_features)[-target_n_features]]

    def transform(self, X, target_n_features=None, **kwargs):
        """
        Shrink a matrix / AnnData object with full markers to the selected markers only.
        If such operation is not supported by your data object,
        you can do it manually using :func:`~UmapL1.get_mask`.

        :param X: Matrix / AnnData to be shrunk
        :param target_n_features: If None, all features with w > 0 are selected. If not None, only select
            `target_n_features` largest features
        :return: Shrunk matrix / Anndata
        """
        return X[:, self.get_mask(target_n_features)]

    def fit_transform(self, X, **kwargs):
        """
        Fit on a matrix / AnnData and then transfer it.

        :param X: The matrix / AnnData to be transformed
        :param kwargs: Other parameters for :func:`UmapL1.fit`.
        :return: Shrunk matrix / Anndata
        """
        return self.fit(X, **kwargs).transform(X)
