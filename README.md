# marker-selection [![Documentation Status](https://readthedocs.org/projects/marker-selection/badge/?version=latest)](https://marker-selection.readthedocs.io/en/latest/?badge=latest) #
Marker Selection by t-SNE and ℓ1-Regularization

## Documentation ##
Detailed documentation is available at https://marker-selection.readthedocs.io/

## Installation ##

requirements: pytorch, scikit-learn, numpy, scipy

recommendations: scanpy, loompy

## Usage ##

### Basics ###
Assuming that you have a dataset in the form of a `scanpy`/`AnnData` object `adata`.

First, import the module:
```python
from compactmarker import TsneL1
```

Then, if you want to train the model with a given strength of l1-regularization:
```python
model = TsneL1(lasso=1e-3).fit(adata.X)
```

Or, if you want to keep a specific number of features:
```python
model_20 = TsneL1.tune(target_n_features=20, X=adata.X)
```
It will perform a binary search on strength of l1-regularization to find the one 
giving desired number of features.

To retain only the selected markers
```python
selected_adata = model.transform(adata)
```

Note that the model has a space complexity of O(n^2), where n is the number of cells. 
Thus, we recommend that you subsample your data to 5,000 to 10,000 cells.
Please refer to "Advanced" section for running on more cells.

### Advanced ###

#### Marker transfering ####
To use one set of markers (e.g., mRNA) to fit the cell-cell similarity defined by another set of markers (e.g., protein).
```python
model.fit(rna_adata.X, X_teacher=protein_adata.X)
```
#### Batch stratification ####
To find markers that are important in multiple samples (batches), you can specify `batches` in `fit()`:
```python
model.fit(rna_adata.X, batches=adata.obs['batch'].values)
```
The dataset will be separated on the batches given, and the loss will be the sum of losses on all separated datasets. In this way, it will not be lured by the markers that separates the markers.

Incidentally, this approach also reduces the memory requirement. If a dataset with n cells is separate into b batches, the space complexity will reduce from O(n^2) to O(b * (n/b)^2) = O(n^2 / b). Thus, if subsampling is not desired, you may randomly separete the dataset into several batches. (That said, do not define the batches as the cell type labels or any category that is biologically meaningful.)

#### Predetemined markers ####
If there are markers you think that should be considered with priority, there are two ways to indicate/enforce it.
1. Use a vector as the parameter `lasso`, and set the corresponding entries to 0. In this way, you remove l1-regularization for that gene.
```python
model = TsneL1(lasso=[0., 0., 1e-5, 1e-5, 1e-5, ...])
model.fit(rna_adata.X)
```
2. Set `must_keep` to nonzero values
```python
model.fit(rna_adata.X, must_keep=[1., 1., 0., 0., 0., ...])
```
If you wish to use both, the lasso parameter should only contain entires whose `must_keep` status is zero. For example:
```python
model = TsneL1(lasso=lasso[must_keep == 0])
model.fit(rna_adata.X, must_keep=must_keep)
```

#### Tuning ####
```python
TsneL1.tune(cls, target_n_features, 
            X=None, X_teacher=None, batches=None, P=None, beta=None, perplexity=30., n_pcs=None, w=None,
            min_lasso=1e-8, max_lasso=1e-2, tolerance=0, smallest_log10_fold_change=0.1, max_iter=100,
            **kwargs)
```

All other parameters of ```compactmarker.TsneL1``` (except for lasso, which is to be tuned) can also be specified.

### Full API ###
Please refer to the [documentation](https://marker-selection.readthedocs.io/).

#### All model parameters ####

- `n_pcs`: If you want to use PCs to calculate the pairwise distances, specify the number of PCs. If you want to use the expression directly, set it to `None`. Default: `None`.
- `w`: Initial value of w. Leaving it as `None` to randomly generate one. Default: `None`.
- `owlqn_history_size`: History size for OWLQN optimization. Set to a smaller value if you encounter an insufficient memory problem. Default: `100`.
- `n_threads`: Number of threads used in calculating pairwise similarity. A linear speed-up is expected so it is recommended to use all CPUs.

## Examples ##

Please refer to `notebooks/` for a few examples.
