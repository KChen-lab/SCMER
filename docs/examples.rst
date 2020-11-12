Examples
=======================

scRNA
-----
The following examples show the basic work flow for RNA datasets.

.. toctree::
   :maxdepth: 2

   melanoma
   pancancer
   ileum-lamina-propria-immunocytes
   bone-marrow
   a549

CyTOF
-----
We also validated SCMER on CyTOF data, where there are far fewer features.

.. toctree::
   :maxdepth: 2

   cytof

CITE-Seq (Genes retaining protein manifold)
-------------------------------------------
SCMER also has the ability to find genes that better represent the manifold defined by proteins.
The genes encoding those proteins are not always the best.

.. toctree::
   :maxdepth: 2

   pbmc-cite-seq

Supplementary
-------------

These notebooks do not use SCMER. They support the validation of the analyses above.
Other files such as gene sets supporting the analyses can be found `here <https://github.com/KChen-lab/marker-selection/tree/master/misc>`_.

.. toctree::
   :maxdepth: 2

   a549-atac-peak
   pbmc-cite-seq-rna

