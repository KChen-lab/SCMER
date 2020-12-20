FAQs
=======================

Installation
------------

Q: I got a OSError [WinError 182] Error loading "...\caffe2_detectron_ops.dll", what can I do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A: Please check your environment (:bash:`conda list intel-openmp`) if :bash:`intel-openmp` is from :bash:`conda-forge` channel.
If so, please reinstall it using :bash:`conda install -c defaults intel-openmp -f`.
