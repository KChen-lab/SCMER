# marker-selection #
Marker Selection by t-SNE and ℓ1-Regularization

## Installation ##

requirements: pytorch, scikit-learn, numpy, scipy

recommendations: scanpy, loompy

## Usage ##

### Basics ###
Assuming that you have a dataset in the form of a `scanpy`/`AnnData` object `adata`.

First, import the module:
```python
import compactmarker._tsne_l1
```

Then, if you want to train the model with a given strength of l1-regularization:
```python
model = compactmarker._tsne_l1.TsneL1(lasso=1e-3).fit(adata.X)
```

Or, if you want to keep a specific number of features:
```python
model_20 = compactmarker._tsne_l1.TsneL1.tune(adata.X, target_n_features=20)
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
To find markers that are important in multiple samples (batches), you can specify `batches` in `fit()`:
```python
model.fit(adata.X, batches=adata.obs['batch'].values)
```
The dataset will be separated on the batches given, and the loss will be the sum of losses on all separated datasets. In this way, it will not be lured by the markers that separates the markers.

Incidentally, this approach also reduces the memory requirement. If a dataset with n cells is separate into b batches, the space complexity will reduce from O(n^2) to O(b * (n/b)^2) = O(n^2 / b). Thus, if subsampling is not desired, you may randomly separete the dataset into several batches. (That said, do not define the batches as the cell type labels or any category that is biologically meaningful.)


Additional Parameters
---------------------
`n_pcs`: If you want to use PCs to calculate the pairwise distances, specify the number of PCs. If you want to use the expression directly, set it to `None`. Default: `None`.

`w`: Initial value of w. Leaving it as `None` to randomly generate one. Default: `None`.

`owlqn_history_size`: History size for OWLQN optimization. Set to a smaller value if you encounter an insufficient memory problem. Default: `100`.

Examples
--------
Please refer to `notebooks/` for a few examples.
