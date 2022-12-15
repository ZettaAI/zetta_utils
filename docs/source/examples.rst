========
Examples
========

Logging
-------

Import ``logger`` object from ``zetta_utils.log`` and use it instead of ``print`` and ``warnings.warn`` statements.

.. code-block:: python

   >>> from zetta_utils.log import logger
   >>> logger.warn("This is a warning")
   >>> logger.info("Info message")
   >>> logger.exception(RuntimeError)


Tensor Operations
-----------------

Generic ops:

.. doctest::

   >>> from zetta_utils import tensor_ops
   >>> import numpy as np
   >>> a = np.ones((2, 2))
   >>> a = tensor_ops.unsqueeze(a)
   >>> print (a.shape)
   (1, 2, 2)


.. doctest::

   >>> from zetta_utils import tensor_ops
   >>> import torch
   >>> t = torch.ones((2, 2))
   >>> t = tensor_ops.unsqueeze(t)
   >>> print (t.shape)
   torch.Size([1, 2, 2])

Generic conversion:

.. doctest::

   >>> from zetta_utils import tensor_ops
   >>> import numpy as np
   >>> a = np.ones((2, 2))
   >>> a = tensor_ops.convert.to_torch(a)
   >>> print (type(a))
   <class 'torch.Tensor'>


.. doctest::

   >>> from zetta_utils import tensor_ops
   >>> import torch
   >>> t = torch.ones((2, 2))
   >>> t = tensor_ops.convert.to_torch(t)
   >>> print (type(t))
   <class 'torch.Tensor'>


BoundingCube
------------

.. doctest::

   >>> from zetta_utils.typing import Vec3D
   >>> from zetta_utils.bcube import BoundingCube
   >>> bcube = BoundingCube.from_coords(
   ...    start_coord=Vec3D(100, 100, 10),
   ...    end_coord=Vec3D(200, 200, 20),
   ...    resolution=Vec3D(4, 4, 40)
   ... )
   >>> print(bcube)
   BoundingBoxND(bounds=((400.0, 800.0), (400.0, 800.0), (400.0, 800.0)), unit='nm')
   >>> slices = bcube.to_slices(resolution=(16, 16, 100))
   >>> print(slices)
   (slice(25, 50, None), slice(25, 50, None), slice(4, 8, None))

Layers
------

Layers for CloudVolume IO:

.. doctest::
   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> # Vanilla CloudVolume Analog
   >>> # Differences with Vanilla CV:
   >>> #   1. Read data type: ``torch.Tensor``.
   >>> #   2. Dimension order: CXYZ
   >>> cvl = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   ... )
   >>> data = cvl[(64, 64, 40), 1000:1100, 1000:1100, 2000:2001]
   >>> data.shape # channel, x, y, z
   torch.Size([1, 100, 100, 1])


   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.typing import Vec3D
   >>> # Advanced features:
   >>> # Custom index resolution, desired resolution, data resolution
   >>> cvl = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm",
   ...    default_desired_resolution=Vec3D(64, 64, 40),
   ...    index_resolution=Vec3D(4, 4, 40),
   ...    data_resolution=Vec3D(128, 128, 40),
   ...    interpolation_mode="img",
   ... )
   >>> data = cvl[16000:17600, 16000:17600, 2000:2001] # (4, 4, 40) indexing
   >>> data.shape # channel, x, y, z
   torch.Size([1, 100, 100, 1])

Layer sets for grouping layers together:

.. doctest::

   >>> from zetta_utils.typing import Vec3D
   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.layer import build_layer_set
   >>> cvl_x0 = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
   ... )
   >>> cvl_x1 = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   ... )
   >>> # Combine the two layers
   >>> lset = build_layer_set(
   ...    layers={"img": cvl_x0, "img_norm": cvl_x1}
   ... )
   >>> # Create an index variable to index both
   >>> idx = (
   ...    Vec3D(64, 64, 40),
   ...    slice(1000, 1100),
   ...    slice(1000, 1100),
   ...    slice(2000, 2001),
   ... )
   >>> data_x0 = lset[Vec3D(64, 64, 40), 1000:1100, 1000:1100, 2000:2001]
   >>> print(list(data_x0.keys()))
   ['img', 'img_norm']
   >>> print(data_x0['img'].shape)
   torch.Size([1, 100, 100, 1])
   >>> # Select read layers as a part of the index
   >>> data_x1 = lset[('img', ), Vec3D(64, 64, 40), 1000:1100, 1000:1100, 2000:2001]
   >>> print(list(data_x1.keys()))
   ['img']


Datasets
--------

You can wrap any layer (include layer set) as a Pytorch dataset.
In this example we will make a dataset out of the followign layer set:

.. doctest::

   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.layer import build_layer_set
   >>> lset = build_layer_set(layers={
   ...    'img': build_cv_layer(path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"),
   ...    'img_norm': build_cv_layer(path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"),
   ... })

To form a layer dataset, we need to specify both the layer and a mapping from sample number to an index that the layer understands.
Such mapping, referred to as sample indexer, will determine what bounding cube is used to fetch training sample #0, #1, etc, as
well as specify how many training samples there will be in total.
In this example, we will be using ``VolumetricStridedIndexer``:

.. doctest::
   >>> from zetta_utils import training
   >>> from zetta_utils.typing import Vec3D
   >>> from zetta_utils.bcube import BoundingCube
   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.layer import build_layer_set
   >>> indexer = training.datasets.sample_indexers.VolumetricStridedIndexer(
   ...    # Range over which to sample
   ...    bcube=BoundingCube.from_coords(
   ...       start_coord=Vec3D(1000, 1000, 2000),
   ...       end_coord=Vec3D(2000, 2000, 2100),
   ...       resolution=Vec3D(64, 64, 40)
   ...    ),
   ...    # How big each chunk will be
   ...    resolution=Vec3D(64, 64, 40),
   ...    chunk_size=Vec3D(128, 128, 1),
   ...    # How close together samples can be
   ...    stride=Vec3D(32, 32, 1),
   ...    # What resolution to get slices at
   ...    index_resolution=Vec3D(64, 64, 40),
   ...    # What to set as `desired_resolution` in the index
   ...    desired_resolution=Vec3D(64, 64, 40),
   ... )
   >>> print(len(indexer)) # total number of samples
   78400
   >>> print(indexer(0))
   (Vec3D(64., 64., 40.), slice(1000, 1128, None), slice(1000, 1128, None), slice(2000, 2001, None))
   >>> print(indexer(1))
   (Vec3D(64., 64., 40.), slice(1032, 1160, None), slice(1000, 1128, None), slice(2000, 2001, None))
   >>> print(indexer(78399))
   (Vec3D(64., 64., 40.), slice(1864, 1992, None), slice(1864, 1992, None), slice(2099, 2100, None))
   >>> dset = training.datasets.LayerDataset(
   ...    layer=lset,
   ...    sample_indexer=indexer,
   ... )
   >>> sample = dset[0]
   >>> print (list(sample.keys()))
   ['img', 'img_norm']
   >>> print (sample['img'].shape)
   torch.Size([1, 128, 128, 1])



Builder
-------

``builder`` provides machinery to represent layers, datasets, or any other registered components
as dictionaries. This can be used to pass in flexible parameters to CLI tools and to allow flexible,
readable specifications of training and inference workflow through ``json``/``yaml``/``cue`` fields.

To make objects of a class buildable with ``builder``:

.. doctest::

   >>> from zetta_utils import builder
   >>> @builder.register("MyClass")
   ... class MyClass:
   ...    def __init__(self, a):
   ...       self.a = a

After an object type is registered, you can represent them as dictionaries by including the matching ``@type`` key
and providing the initialization parameters:

.. doctest::

   >>> spec = {
   ...    "@type": "MyClass",
   ...    "a": 100
   ... }
   >>> obj = builder.build(spec)
   >>> print (type(obj))
   <class 'MyClass'>
   >>> print (obj.a)
   100

All user-facing ``zetta_utils`` objects are registered with ``builder`` on module import.
Don't forget to import all ``zetta_utils`` modules that you want the builder to know about.
You can check out the state of the current registry by inspecting ``builder.REGISTRY``

``builder`` will build your objects recursively. That means that you can specify complex structures,
such as the dataset from the earlier example:

.. doctest::

   >>> from zetta_utils import builder
   >>> from zetta_utils import  training
   >>> spec = {
   ...    "@type": "LayerDataset",
   ...    "layer": {
   ...       "@type": "build_layer_set",
   ...       "layers": {
   ...          "img": {"@type": "build_cv_layer", "path": "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"},
   ...          "img_norm": {"@type": "build_cv_layer", "path": "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"}
   ...       }
   ...    },
   ...    "sample_indexer": {
   ...        "@type": "VolumetricStridedIndexer",
   ...        "bcube": {
   ...           "@type": "BoundingCube",
   ...           "start_coord": (1000, 1000, 2000),
   ...           "end_coord": (2000, 2000, 2100),
   ...           "resolution": (64, 64, 40),
   ...        },
   ...        "resolution": (64, 64, 40),
   ...        "chunk_size": (128, 128, 1),
   ...        "stride": (32, 32, 1),
   ...        "index_resolution": (64, 64, 40),
   ...        "desired_resolution": (64, 64, 40),
   ...    }
   ... }
   >>> dset = builder.build(spec)
   >>> sample = dset[0]
   >>> print (list(sample.keys()))
   ['img', 'img_norm']
   >>> print (sample['img'].shape)
   torch.Size([1, 128, 128, 1])
