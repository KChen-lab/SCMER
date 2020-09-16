# marker-selection
Marker Selection by t-SNE and ℓ1-Regularization

Installation
--------------

requirements: pytorch

recommendations: scanpy, loompy

Usage
-----
First, import the module:
```python
import compactmarker._tsne_l1
```

Then, if you want to train the model with a given strength of l1-regularization:
```python
model = compactmarker._tsne_l1.TsneL1(lasso=1e-3)
model.fit(adata.X)
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

Additional Parameters
---------------------
`n_pcs`: If you want to use PCs to calculate the pairwise distances, specify the number of PCs. If you want to use the expression directly, set it to `None`. Default: `None`.
`w`: Initial value of w. Leaving it as `None` to randomly generate one. Default: `None`.
`owlqn_history_size`: History size for OWLQN optimization. Set to a smaller value if you encounter an insufficient memory problem. Default: `100`.
