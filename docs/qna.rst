FAQs
=======================

Installation
------------

Q: I got a OSError [WinError 182] Error loading "...\caffe2_detectron_ops.dll". What can I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: Please check your environment (:bash:`conda list intel-openmp`) if :bash:`intel-openmp` is from :bash:`conda-forge` channel.
If so, please reinstall it using :bash:`conda install -c defaults intel-openmp -f`.

Q: I got an error for importing `tables`. What can I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A: Running `conda install snappy` seems to have solved this issue, although the reason is unknown.
Ref: https://stackoverflow.com/questions/63022939/having-trouble-loading-tables-in-a-conda-environment-after-an-apparently-sucessf
