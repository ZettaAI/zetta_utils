.. module:: zetta_utils

Welcome to ``zetta_utils`` documentation!
=========================================

.. include:: ../../README.rst
  :start-after: teaser-begin
  :end-before: teaser-end

..
   **zetta_utils** is a colleciton of core components used in Zetta AI's connectomics pipeline.



Getting Started
===============

..
  _`hosted on PyPI <https://pypi.org/project/>`_.

The recommended installation method is `pip <https://pip.pypa.io/en/stable/>`_-installing into a `virtualenv <https://hynek.me/articles/virtualenv-lives/>`_:

.. code-block:: console

   $ git clone git@github.com:ZettaAI/zetta_utils.git
   $ cd zetta_utils
   $ pip install '.[all]'

Note that the command above would install all of the optional ``zetta_utils`` modules.
In order to get a more barebones installation, you can specify exact modules you require in comma separated format inside brackets.
You can refer to ``pyproject.toml`` ``project.optional-dependencies`` section for the full list of the optional modules.

.. note::

   If you are performing a local editable install(``pip install -e .[{modules}]``), you may want to set environment variable ``SETUPTOOLS_ENABLE_FEATURES=legacy-editable``.
   This is caused by a shortcoming of ``setuptools`` (https://github.com/pypa/setuptools/issues/3535).

If you are planning to use `zetta_utils.parsing.cue` or `zetta_utils.cli`, you will need to install `cuelang <https://cuelang.org/>`_.
It is a simple two-step process which is described in detail int their `Documentation <https://cuelang.org/docs/install/>`_.

If you are planning to use `zetta_utils.viz` toolkit, you will need to install nodejs:

.. code-block:: console

    $ curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    $ sudo apt install -y nodejs

Depending on your system, you may need to install severqal other non-python dependencies. Please refer to the provided `Dockerfiles <https://github.com/ZettaAI/zetta_utils/tree/main/docker>`_ for the comprehensive setup process.

The next steps will get you up and running in no time:

- `examples` will give you a comprehensive tour of ``zetta_utils``'s features.
- **Module documentation** will show you how each individual module is intended to be used.
- `developer_guide` will give you all of the information necessary to contribute to ``zetta_utils``.
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
