# SCMER - Manifold Preserving Feature Selection 
[![Documentation Status](https://readthedocs.org/projects/scmer/badge/?version=latest)](https://scmer.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/scmer?color=blue&logo=pypi)](https://pypi.org/project/scmer) ![PyPI - Downloads](https://img.shields.io/pypi/dm/scmer) [![Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://doi.org/10.24433/CO.6781338.v1)

SCMER is a feature selection methods designed for single-cell data analysis. 
It selects a compact sets of markers that preserve the manifold in the original data.
It can also be used for data integration by using features in one modality to match the manifold of another modality.

## Tutorials ##
Tutorials are available at https://scmer.readthedocs.io/en/latest/examples.html

You may start with the [Melanoma data (Tiorsh et al.)](https://scmer.readthedocs.io/en/latest/melanoma.html).

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

## Contact ##
I do monitor the "Issues" and aim to clear any issues in a few weeks.
If you have an urgent request, please email `shaohengliang@gmail.com`.

## Version log ##
- 0.1.1 (6/8/2023) Fixed an issue that caused an error when `np.matrix` is used instead of `np.array`.
- 0.1.0a4 (2/12/2023) Fixed an issue that caused an error when batch correction is enabled on GPU runs. CPU runs were not affected. 
- 0.1.0a3 (2/17/2021) Initial version.
