========
Examples
========

Logging
-------

Import ``logger`` object from ``zetta_utils.log`` and use it instead of ``print`` and ``warnings.warn`` statements.

.. doctest::

   >>> from zetta_utils.log import logger
   >>> logger.warn("This is a warning")
   >>> logger.info("Info message")
   >>> logger.exception(RuntimeError)


Tensor Operations
-----------------

Generic ops:

.. doctest::

   >>> import numpy as np
   >>> a = np.ones((2, 2))
   >>> a = zu.tensor.ops.unsqueeze(a)
   >>> print (a.shape)
   (1, 2, 2)


.. doctest::

   >>> import torch
   >>> t = torch.ones((2, 2))
   >>> t = zu.tensor.ops.unsqueeze(t)
   >>> print (t.shape)
   torch.Size([1, 2, 2])

Generic conversion:

.. doctest::

   >>> import numpy as np
   >>> a = np.ones((2, 2))
   >>> a = zu.tensor.convert.to_torch(a)
   >>> print (type(a))
   <class 'torch.Tensor'>


.. doctest::

   >>> import torch
   >>> t = torch.ones((2, 2))
   >>> t = zu.tensor.convert.to_torch(t)
   >>> print (type(t))
   <class 'torch.Tensor'>


BoundingCube
------------

.. doctest::

   >>> bcube = zu.bbox.BoundingCube.from_coords(
   ...    start_coord=(100, 100, 10),
   ...    end_coord=(200, 200, 20),
   ...    resolution=(4, 4, 40)
   ... )
   >>> print(bcube)
   BoundingBoxND(bounds=((400, 800), (400, 800), (400, 800)), unit='nm')
   >>> slices = bcube.to_slices(resolution=(16, 16, 100))
   >>> print(slices)
   (slice(25, 50, None), slice(25, 50, None), slice(4, 8, None))

Layers
------

Layers for CloudVolume IO:

.. doctest::

   >>> # Vanilla CloudVolume Analogue
   >>> cvl = zu.io.build_cv_layer(
   ...    path="gs://fafb_v15_aligned/v0/img/img_norm"
   ... )
   >>> data = cvl[(64, 64, 40), 1000:1100, 1000:1100, 2000:2001]
   >>> data.shape # batch, channel, x, y, z
   (1, 1, 100, 100, 1)


   >>> # Custom index resolution, desired resolution, data resolution
   >>> cvl = zu.io.build_cv_layer(
   ...    path="gs://fafb_v15_aligned/v0/img/img_norm",
   ...    default_desired_resolution=(64, 64, 40),
   ...    index_resolution=(4, 4, 40),
   ...    data_resolution=(128, 128, 40),
   ...    interpolation_mode="img",
   ... )
   >>> data = cvl[16000:17600, 16000:17600, 2000:2001] # (4, 4, 40) indexing
   >>> data.shape # batch, channel, x, y, z
   (1, 1, 100, 100, 1)

Layer sets for grouping layers together:

.. doctest::

   >>> cvl_x0 = zu.io.build_cv_layer(
   ...    path="gs://fafb_v15_aligned/v0/img/img"
   ... )
   >>> cvl_x1 = zu.io.build_cv_layer(
   ...    path="gs://fafb_v15_aligned/v0/img/img_norm"
   ... )
   >>> # Combine the two layers
   >>> lset = zu.io.build_layer_set(
   ...    layers={"img": cvl_x0, "img_norm": cvl_x1}
   ... )
   >>> # Create an index variable to index both
   >>> idx = (
   ...    (64, 64, 40),
   ...    slice(1000, 1100),
   ...    slice(1000, 1100),
   ...    slice(2000, 2001),
   ... )
   >>> data_x0 = lset[(64, 64, 40), 1000:1100, 1000:1100, 2000:2001]
   >>> print(list(data_x0.keys()))
   ['img', 'img_norm']
   >>> print(data_x0['img'].shape)
   (1, 1, 100, 100, 1)
   >>> # Select read layers as a part of the index
   >>> data_x1 = lset[('img', ), (64, 64, 40), 1000:1100, 1000:1100, 2000:2001]
   >>> print(list(data_x1.keys()))
   ['img']


Datasets
--------

You can wrap any layer (include layer set, which is also a laywer) as a Pytorch dataset.
In this example we will make a dataset out of a simple 2-layer layer set:

.. doctest::

   >>> lset = zu.io.build_layer_set(layers={
   ...    'img': zu.io.build_cv_layer(path="gs://fafb_v15_aligned/v0/img/img"),
   ...    'img_norm': zu.io.build_cv_layer(path="gs://fafb_v15_aligned/v0/img/img_norm"),
   ... })

Now that we have the layer that will serve as the basis for our datast, we need to specify how each sample index number,
which is an integer, will be mapped to an index type that our layer understands, which in this case is a volumetric
index. As this behaviour can be parametrized in many ways, it is represented by a custom indexer object that performs the mapping.
In this example, we will be using ``VolumetricStepIndexer``:

.. doctest::

   >>> indexer = zu.training.datasets.sample_indexers.VolumetricStepIndexer(
   ...    # Range over which to sample
   ...    bcube=zu.bbox.BoundingCube.from_coords(
   ...       start_coord=(1000, 1000, 2000),
   ...       end_coord=(2000, 2000, 2100),
   ...       resolution=(64, 64, 40)
   ...    ),
   ...    # How big each sample will be
   ...    sample_size=(128, 128, 1),
   ...    sample_size_resolution=(64, 64, 40),
   ...    # How close together samples can be
   ...    step_size=(32, 32, 1),
   ...    step_size_resolution=(64, 64, 40),
   ...    # What resolution to get slices at
   ...    index_resolution=(64, 64, 40),
   ...    # What to set as `desired_resolution` in the index
   ...    desired_resolution=(64, 64, 40),
   ... )
   >>> print(len(indexer)) # total number of samples
   78400
   >>> print(indexer(0))
   ((64, 64, 40), slice(1000, 1128, None), slice(1000, 1128, None), slice(2000, 2001, None))
   >>> print(indexer(1))
   ((64, 64, 40), slice(1032, 1160, None), slice(1000, 1128, None), slice(2000, 2001, None))
   >>> print(indexer(78399))
   ((64, 64, 40), slice(1864, 1992, None), slice(1864, 1992, None), slice(2099, 2100, None))

.. doctest::

   >>> dset = zu.training.datasets.LayerDataset(
   ...    layer=lset,
   ...    sample_indexer=indexer,
   ... )
   >>> sample = dset[0]
   >>> print (list(sample.keys()))
   ['img', 'img_norm']
   >>> print (sample['img'].shape)
   torch.Size([1, 1, 128, 128, 1])



Builder
-------

``zu.builder`` provides machinery to represent layers, datasets, or any other registered components
as dictionaries. This can be used to pass in flexible parameters to CLI tools and to allow flexible,
readable specifications of training and inference workflow through ``json``/``yaml``/``cue`` fiels.

To make objects of a class buildable with ``zu.builder``:

.. doctest::

   >>> @zu.builder.register("MyClass")
   ... class MyClass:
   ...    def __init__(self, a):
   ...       self.a = a
   >>> spec = {
   ...    "<type>": "MyClass",
   ...    "a": 100
   ... }
   >>> obj = zu.builder.build(spec)
   >>> print (type(obj))
   <class 'MyClass'>
   >>> print (obj.a)
   100

User-facing ``zetta_utils`` objects are all registered with ``zu.builder``. You can check out the state of the current registry
by inspecting ``zu.builder.REGISTRY``

``zu.builder`` will build your objects recursively. That means that you can specify complex structures,
such as the dataset from earlier example:

.. doctest::

   >>> spec = {
   ...    "<type>": "LayerDataset",
   ...    "layer": {
   ...       "<type>": "LayerSet",
   ...       "layers": {
   ...          "img": {"<type>": "CVLayer", "path": "gs://fafb_v15_aligned/v0/img/img"},
   ...          "img_norm": {"<type>": "CVLayer", "path": "gs://fafb_v15_aligned/v0/img/img_norm"}
   ...       }
   ...    },
   ...    "sample_indexer": {
   ...        "<type>": "VolumetricStepIndexer",
   ...        "bcube": {
   ...           "<type>": "BoundingCube",
   ...           "start_coord": (1000, 1000, 2000),
   ...           "end_coord": (2000, 2000, 2100),
   ...           "resolution": (64, 64, 40),
   ...        },
   ...        "sample_size": (128, 128, 1),
   ...        "sample_size_resolution": (64, 64, 40),
   ...        "step_size": (32, 32, 1),
   ...        "step_size_resolution": (64, 64, 40),
   ...        "index_resolution": (64, 64, 40),
   ...        "desired_resolution": (64, 64, 40),
   ...    }
   ... }
   >>> dset = zu.builder.build(spec)
   >>> sample = dset[0]
   >>> print (list(sample.keys()))
   ['img', 'img_norm']
   >>> print (sample['img'].shape)
   torch.Size([1, 1, 128, 128, 1])

.. note::

   **What are the advantages of builder based specs?**
   Comparing to building custom parsers for each project/workflow, builder based specs reduce code duplication,
   remove the need to update parser logic when object representations change, and can handle difficulty in dealing with nested structures.
   * *Over python based specs*: Absense of loops and conditionals makes
   *
