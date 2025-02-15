# SCMER - Manifold Preserving Feature Selection 
[![Documentation Status](https://readthedocs.org/projects/scmer/badge/?version=latest)](https://scmer.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/scmer?color=blue&logo=pypi)](https://pypi.org/project/scmer) ![PyPI - Downloads](https://img.shields.io/pypi/dm/scmer) [![Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://doi.org/10.24433/CO.6781338.v1)

SCMER is a feature selection methods designed for single-cell data analysis. 
It selects a compact sets of markers that preserve the manifold in the original data.
It can also be used for data integration by using features in one modality to match the manifold of another modality.

## Contact ##
Please use the "Issues" to submit a question. You should get a reply in a few days. If not, please don't hesitate to email `shaohengliang@gmail.com` to bring it to my attention.

## Tutorials ##
Tutorials are available at https://scmer.readthedocs.io/en/latest/examples.html

You may start with the [Melanoma data (Tiorsh et al.)](https://scmer.readthedocs.io/en/latest/melanoma.html).


### Installation ###
The latest version is 0.1.3 that can be installed through PyPI. For example,
```bash
conda create -n py311_scmer python=3.11 jupyter git
conda activate py311_scmer
conda install -c conda-forge scanpy python-igraph leidenalg
pip install git+https://github.com/KChen-lab/SCMER
git clone https://github.com/KChen-lab/SCMER
```
To add GPU support, install a proper version of [PyTorch](https://pytorch.org/get-started/locally/). For example,
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Long Term Support ###
We try to keep the package work with new versions of Python and other dependencies. 
- Lastest tested version: python 3.11 + torch 2.1.1 + SCANPY 1.9.6
- OS: Windows 10.
- Hardware: AMD Ryzen R5 3600 + Nvidia RTX 3080

For other tested version, please check the `lts` folder in this repository.
The four scripts in the folder can also give you an idea of how to run SCMER for a given scenario.

### Running in R ###
Python is more established in machine learning tasks, but many people prefer R as their primary language in data science. Luckily, a thin wrapper is what you need to run SCMER in R. Please see this short [tutorial](https://htmlpreview.github.io/?https://github.com/KChen-lab/SCMER/blob/master/notebooks/melanoma-gpu-with-batch-in-r.nb.html).

## Full Documentation ##
Detailed documentation is available at https://scmer.readthedocs.io/en/latest/

The mechanism and capabilities of SCMER is detailed in our pre-print [Single-Cell Manifold Preserving Feature Selection (SCMER)](https://www.biorxiv.org/content/10.1101/2020.12.01.407262v1)



## Additional package info
Using GPU can be tricky sometimes. Here is a [list of package versions](https://github.com/KChen-lab/SCMER/blob/master/notebooks/package_versions.txt) we successfully used with GPU.

## Publication ##
Single-cell manifold-preserving feature selection for detecting rare cell populations *Nature Computational Science* (2021)
- Paid access: https://www.nature.com/articles/s43588-021-00070-7
- Free access (no download): https://rdcu.be/ckZGT
- BioRxiv preprint: https://www.biorxiv.org/content/10.1101/2020.12.01.407262v1.full

## Version log ##
- 0.1.3 (12/8/2023) SCMER now compatible with Python 3.11 + torch 2.1.1 + SCANPY 1.9.6.
- 0.1.1 (6/8/2023) Fixed an issue that caused an error when `np.matrix` is used instead of `np.array`.
- 0.1.0a4 (2/12/2023) Fixed an issue that caused an error when batch correction is enabled on GPU runs. CPU runs were not affected. 
- 0.1.0a3 (2/17/2021) Initial version.
