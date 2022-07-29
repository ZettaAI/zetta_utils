.. module:: zetta_utils

Welcome to ``zetta_utils`` documentation!
=======================================

.. include:: ../README.rst
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

``zetta_ai`` is a Python-only package.
The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_:

.. code-block:: console

   $ git clone git@github.com:ZettaAI/zetta_utils.git
   $ cd zetta_utils
   $ pip install -e .

The next steps will get you up and running in no time:

- `overview` will show you a simple example of ``zetta_utils`` in action and introduce you to its philosophy.
  Afterwards, you can start incorporating ``zetta_utils`` components into your Connectomics workflows
  and understand what drives ``zetta_utils``'s design.
- `examples` will give you a comprehensive tour of ``zetta_utils``'s features.
  After reading, you will know about our advanced features and how to use them.
- `developer_guide` will give you all of the information necessary to contribute to ``zetta_utils``.
- If at any point you get confused by some terminology, please check out our `glossary`.


Day-to-Day Usage
================

- `zetta_utils.log` provides a well formated disk backed logging system.

- `zetta_utils.widgets` provides nifty visualization tools for `Jupyter <https://jupyter.org/>`_.

- `zetta_utils.tensor` is a unified set of operations that support *both* ``np.ndarray`` and ``torch.Tensor`` tensor types.
  Use ``zetta_utils.tensor.ops.unsqueeze(t)`` without worrying which type ``t`` is.

- `zetta_utils.bbox` is generalizable implementation of N-dimensional bounding boxes that support custom resoluiton at each of the axis.

- `zetta_utils.io` is flexible toolkit for data IO from diverse backends suitable for both training and inference.

- `zetta_utils.training` includes tools such as CloudVolume based training datasets, Pytorch Lightning integration (WIP) and more.

- `zetta_utils.builder` is utility for building python objects and workflows from nested dictionaries. It is an
  improved extension of `procspec <https://github.com/seunglab/procspec>`_ and
  `artificery <https://github.com/seung-lab/artificery>`_ packages previously used by SeungLab.


.. include:: ../README.rst
   :start-after: -project-information-

----


Full Table of Contents
======================

.. toctree::
  :maxdepth: 2
  :caption: Contents:

  overview
  examples
  zetta_utils.log
  zetta_utils.widgets
  zetta_utils.tensor
  zetta_utils.bbox
  zetta_utils.io
  zetta_utils.training
  zetta_utils.builder
  developer_guide

.. toctree::
  :maxdepth: 1

  license
