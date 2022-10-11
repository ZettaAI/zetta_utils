.. module:: zetta_utils

Welcome to ``zetta_utils`` documentation!
=========================================

.. include:: ../../README.rst
  :start-after: teaser-begin
  :end-before: teaser-end

..
   **zetta_utils** is a colleciton of core components used in Zetta AI's connectomics pipeline.

.. note::

   This project is in pre-release stage and the APIs are subject to change.


Getting Started
===============

..
  _`hosted on PyPI <https://pypi.org/project/>`_.

The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_:

.. code-block:: console

   $ git clone git@github.com:ZettaAI/zetta_utils.git
   $ cd zetta_utils
   $ pip install -e .


If you are planning to use `zetta_utils.parsing.cue` or `zetta_utils.cli`, you will need to install `cuelang <https://cuelang.org/>`_.
It is a simple two-step process which is described in detail int their `Documentation <https://cuelang.org/docs/install/>`_.

If you are planning to use `zetta_utils.viz` toolkit, you will need to install nodejs:

.. code-block:: console

    $ curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    $ sudo apt install -y nodejs


The next steps will get you up and running in no time:

- `examples` will give you a comprehensive tour of ``zetta_utils``'s features.
- **Module documentation** will show you how each individual module is intended to be used.
- `developer_guide` will give you all of the information necessary to contribute to ``zetta_utils``.
- If at any point you get confused by some terminology, please check out our `glossary`.


Day-to-Day Usage
================

- `zetta_utils.tensor` is a unified set of operations that support *both* ``np.ndarray`` and ``torch.Tensor`` tensor types.
  Use ``zetta_utils.tensor_ops.unsqueeze(t)`` without worrying which type ``t`` is.

- `zetta_utils.bbox` is generalizable implementation of N-dimensional bounding boxes that support custom resoluiton at each of the axis.

- `zetta_utils.layer` is flexible abstraction for data IO from diverse backends suitable for both training and inference.

- `zetta_utils.training` includes tools such as CloudVolume based training datasets, Pytorch Lightning integration (WIP) and more.

- `zetta_utils.log` provides a well formated disk backed logging system.

- `zetta_utils.viz` provides nifty visualization tools for `Jupyter <https://jupyter.org/>`_.

- `zetta_utils.builder` is utility for building python objects and workflows from nested dictionaries. It is an
  improved extension of `procspec <https://github.com/seunglab/procspec>`_ and
  `artificery <https://github.com/seung-lab/artificery>`_ packages previously used by SeungLab.


.. include:: ../../README.rst
   :start-after: project-info-begin
   :end-before: project-info-end

----


Full Table of Contents
======================

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  examples
  built_in_components
  modules
  developer_guide
  glossary

.. toctree::
  :maxdepth: 1

  license
