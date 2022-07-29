Welcome to `zetta_utils` documentation!
===================================

`zetta_utils` is a colleciton of core components used in Zetta AI's connectomics pipeline.
It consists of multplie largely indendent componets, including:

* `spec_parser`: A utility for building python objects from dictionaries. It is an
  improved extension of `procspec <https://github.com/seunglab/procspec>`_ and
  `artificery <https://github.com/seung-lab/artificery>`_ packages previously used by SeungLab.

* `tensor`: A unified set of operations each of which supports _both_ np.ndarray and torch.Tensor types.

* `io`: A toolkit for flexible data IO from diverse backends, suitable for both training and inference.

* `training`:


.. note::

   This project is under active development, so some of the APIs are subject to change.

Contents
--------

.. toctree::

   usage
   api
