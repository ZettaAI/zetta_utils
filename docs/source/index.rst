.. module:: zetta_utils

Welcome to ``zetta_utils`` documentation!
=========================================

.. include:: ../../README.rst
  :start-after: teaser-begin
  :end-before: teaser-end

..
   **zetta_utils** is a collection of core components used in Zetta AI's connectomics pipeline.



Getting Started
===============

..
  _`hosted on PyPI <https://pypi.org/project/>`_.

The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_:

.. code-block:: console

   $ git clone  --recurse-submodules git@github.com:ZettaAI/zetta_utils.git
   $ cd zetta_utils
   $ ./install_zutils.py


If you want to let zetta_utils generate **neuroglancer** links automatically, you will need to set:

.. code-block:: console

   NG_STATE_SERVER_TOKEN


If you want to use **Grafana** integration for logging, you will need to set:

.. code-block:: console

   GRAFANA_CLOUD_ACCESS_KEY

Please consult the `Grafana API documentation <https://grafana.com/docs/grafana-cloud/reference/create-api-key/>`_ for how to set up an API key.

To use **KEDA** (Kubernetes Event-driven Autoscaling), you will first need to install `helm <https://helm.sh/docs/intro/install/>`_.


If you are planning to use `zetta_utils.viz` toolkit, you will need to install **nodejs**:

.. code-block:: console

    $ curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    $ sudo apt install -y nodejs


The next steps will get you up and running in no time:

- `subchunkable_apply_flow_quick_start_guide` will introduce you to handling volumetric data in ``zetta_utils`` CUE files and running simple tasks using SubchunkableApplyFlow.
- `examples` will give you a comprehensive tour of ``zetta_utils``'s features.
- **Module documentation** will show you how each individual module is intended to be used.
- `developer_guide` will give you all of the information necessary to start contributing to ``zetta_utils``.
- If at any point you get confused by some terminology, please check out our `glossary`.


Day-to-Day Usage
================

- `zetta_utils.log` provides a well formated disk backed logging system.

- `zetta_utils.tensor` is a unified set of operations that support *both* ``np.ndarray`` and ``torch.Tensor`` tensor types.
  Use ``zetta_utils.tensor_ops.unsqueeze(t)`` without worrying which type ``t`` is.

- `zetta_utils.layer` offers flexible abstraction for data IO.

- `zetta_utils.training` set of tools and integrations for neural net training.

- `zetta_utils.viz` provides nifty visualization tools for `Jupyter <https://jupyter.org/>`_.

- `zetta_utils.builder` is utility for building python objects and workflows from nested dictionaries.

- `subchunkable_apply_flow` is an inference Flow that supports arbitrary subchunking, cropping, and blending.

.. include:: ../../README.rst
   :start-after: project-info-begin
   :end-before: project-info-end

----


Full Table of Contents
======================

.. toctree::
  :maxdepth: 2
  :caption: Contents

  subchunkable_apply_flow_quick_start_guide
  examples
  built_in_components
  modules
  developer_guide
  glossary

.. toctree::
  :maxdepth: 1

  license
