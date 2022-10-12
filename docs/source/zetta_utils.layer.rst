``zetta_utils.layer``
=====================

Description
-----------

``zetta_utils.layer`` is an abstraction that allows reading form and writing to external data sources.
Users interact with external data sources by creating data layers (``zetta_utils.layer.Layer``).
A layer represents an indexable set of data, and is defined by an indexing scheme, IO backend,
and a set of processors applied to the data/indexes during IO operations.

Indexing scheme defines what information will be passed to the IO backend and specifies
how to translate user index inputs to a standardized format.
IO backend defines how to retrieve the data given an index.
IO backends may be used to support various data formats (eg CloudVolume, h5, DynamoDB),
but also to achieve more interesting behaviours, such as grouping layers together,
mapping index ranges to different data sources, etc.

Shortcuts for commonly used layer setups can be created
For example, ``zetta_utils.cloudvol.build_cv_layer(...)`` will build a volumetric-indexed layer with a
CloudVolume backend that includes processors that allow the user to index the layer at arbitrary resolutions.

Available Layer Setups
----------------------

.. autofunction:: zetta_utils.layer.build_layer_set

.. autofunction:: zetta_utils.cloudvol.build_cv_layer
