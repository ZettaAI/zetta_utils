``zetta_utils.io``
==================

``zetta_utils.io`` module handles reading form and writing to external data sources.
Users interact with external data sources by creating data layers (``zetta_utils.io.Layer``).
A layer represents an indexable set of data, and is defined by an indexing scheme, IO backend,
and a set of processors applied to the data/indexes during IO operations.

Indexing scheme defines what information will be passed to the IO backend and specifies
how to translate user index inputs to a standardized format.
IO backend defines how to retrieve the data given an index.
IO backends may be used to support various data formats (eg CloudVolume, h5, DynamoDB),
but also to achieve more interesting behaviours, such as grouping layers together,
mapping index ranges to different data sources, etc.

Users may customize their own layer setups, or use one of the pre-defined structures from ``zetta_utils.io.layer.shortcuts``.
For example, ``zetta_utils.io.layer.shortcuts.build_cv_layer(...)`` will help a user build a volumetric-indexed layer with a
CloudVolume backend that includes processors that allow the user to index the layer at arbitrary resolutions.
Advanages of using such a layer relative to a vanilla CloudVolume are showcased in TODO.

.. toctree::
  :maxdepth: 2
  :caption: Layer Shortcuts:
