=================
SubchunkableApplyFlow Quick Start Guide
=================

Introduction
============

This is the quick start guide for running inference in ``zetta_utils``, intended to be a crash-course / walkthrough for configuring and running tasks using ``SubchunkableApplyFlow``. This guide will introduce three basic concepts of ``zetta_utils`` and highlight some cool features without going into too much detail:

#. **VolumetricLayer**
#. **Builder and CUE files**
#. **SubchunkableApplyFlow**

.. note::

  If you see a **Collapsed Section** like the one that follows, you can safely ignore it on your first read. The notes contain additional details about the design and usage of the concepts introduced in this guide, intended to be an introduction for advanced users and developers.

.. collapse:: Example of extra information

   Information that is nitty-gritty.

      >>> print("example usage")
      example usage

VolumetricLayer
===============

VolumetricLayer Basics
----------------------

``zetta_utils`` uses **VolumetricLayer** to handle volumetric data. You can think of ``VolumetricLayer`` as a CXYZ PyTorch tensor that:

*  includes multiple resolutions, both from existing MIPs and through on-demand downsampling
*  supports arbitrary index, reading, writing processors (more on this later)
*  is backed by a persistent, chunked storage on a file system (possibly compressed), whether local or remote, with in-memory caching.

For this guide, we will use FAFB v15, which is a ``precomputed`` dataset, and use the CloudVolume backend. You can use ``build_cv_layer`` to build a ``VolumetricLayer``, and index into it using the desired resolution and a slice:

.. doctest::

   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.geometry import Vec3D
   >>> # Vanilla CloudVolume Analog
   >>> # Differences with Vanilla CV:
   >>> #   1. Read data type: ``torch.Tensor``.
   >>> #   2. Dimension order: CXYZ
   >>> cvl = build_cv_layer(
   ...     path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   ... )
   >>> data = cvl[Vec3D(64, 64, 40), 13000:13100, 4000:4100, 2000:2001]
   >>> data.shape # channel, x, y, z
   (1, 100, 100, 1)

.. collapse:: Vec3D

   **Vec3D** (and its covariant subtype ``IntVec3D = Vec3D[int]``) is a convenient container for 3 dimensional vectors that supports a variety of (statically) type-inferenced dunder (double-underscore) methods and operations. For instance,

      >>> Vec3D(1, 2, 3) * Vec3D(2, 3, 4)
      Vec3D(2, 6, 12)
      >>> Vec3D(1, 2, 3) / Vec3D(3, 2, 1)
      Vec3D(0.3333333333333333, 1.0, 3.0)
      >>> Vec3D(1, 2, 3) // Vec3D(3, 2, 1)
      Vec3D(0, 1, 3)
      >>> Vec3D(1, 2, 3) % Vec3D(3, 2, 1)
      Vec3D(1, 0, 0)

One neat feature of ``VolumetricLayer`` is that the resolutions

* user expects for the output
* used to index into the data
* data is stored at

can be specified *independently*.

For instance, suppose that you were looking at the FAFB v15 in neuroglancer, and decided that you wanted to see what a given area looked like at (192, 192, 80) nm resolution. You only know the coordinates in (4, 4, 40) nm resolution since that's what neuroglancer displays, and you also know that the data only exists at (64, 64, 40) nm resolution. You can initialise the ``VolumetricLayer`` with the kwargs ``default_desired_resolution``, ``index_resolution``, ``data_resolution`` respectively to get the data on-the-fly at the resolution you want.

.. doctest::

   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.geometry import Vec3D
   >>> # Advanced features:
   >>> # Custom index resolution, desired resolution, data resolution
   >>> cvl = build_cv_layer(
   ...     path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm",
   ...     default_desired_resolution=Vec3D(192, 192, 80),
   ...     index_resolution=Vec3D(4, 4, 40),
   ...     data_resolution=Vec3D(64, 64, 40),
   ...     interpolation_mode="img",
   ... )
   >>> data = cvl[211200:216000, 64800:69600, 2000:2002] # (4, 4, 40) indexing
   >>> data.shape # channel, x, y, z at (192, 192, 80) resolution
   (1, 100, 100, 1)

This feature can be used to:

* apply masks that are in one resolution to image data in a different resolution without running downsampling / upsampling separately
* cut down on data egress costs if a task is set up to use high resolution data but low resolution is sufficient
* simultaneously index into multiple layers (using ``VolumetricLayerSet``) that are all at different resolutions and return a dictionary of cutouts with the same shape.

.. note::

   ``interpolation_mode`` can be ``img``, ``field``, ``mask``, or ``segmentation``; this specifies the algorithm to use during interpolation.


VolumetricIndex
---------------

In the previous subsection, we directly used the resolution and slice to specify the ROI (region of interest) for reading a given ``VolumetricLayer``. While this is sufficient for just reading the data in a given area, there are many operations that we would like to do on a specified ROI: we may wish to crop or pad the ROI, expand and snap a given ROI to a grid with a given offset and grid size, come up with the intersection of two ROIs, and so forth.

``zetta_utils`` provides **VolumetricIndex** for this purpose. Let's define a (5, 7, 11) pixel ROI aligned to the origin, at (4, 4, 30) nm resolution:

.. doctest::

   >>> from zetta_utils.layer.volumetric import VolumetricIndex
   >>> from zetta_utils.geometry import Vec3D
   >>> idx = VolumetricIndex.from_coords(
   ...     start_coord = Vec3D(0, 0, 0),
   ...     end_coord = Vec3D(5, 7, 11),
   ...     resolution = Vec3D(4, 4, 30)
   ... )
   >>> idx
   VolumetricIndex(resolution=Vec3D(4, 4, 30), bbox=BBox3D(bounds=((0, 20), (0, 28), (0, 330)), unit='nm', pprint_px_resolution=(1, 1, 1)), chunk_id=0, allow_slice_rounding=False)
   >>> print(idx.pformat())
   (0.0, 0.0, 0.0) - (5.0, 7.0, 11.0)
   >>> idx.shape
   Vec3D(5, 7, 11)

As you can see, ``VolumetricIndex.from_coords`` has automatically calculated the start and end coordinates in physical space from the provided resolution. The ``VolumetricIndex`` also carries a ``chunk_id``, which is a unique integer that is assigned sequentially to the task indices during some operations (including subchunkable). This is unused in most cases, though, and defaults to 0.

.. collapse:: BBox3D

   **BBox3D** is the class that powers ``VolumetricIndex``; the only difference between the two is that ``BBox3D`` is a cuboid in space without any resolution data attached to it, while ``VolumetricIndex`` has a resolution. Internally, most of the methods in ``VolumetricIndex`` are just delegated to the methods of the same name in ``BBox3D`` with the resolution. You should not have to interact with ``BBox3D`` very much.

Let's try padding and cropping our new ``VolumetricIndex``:


.. doctest::

   >>> idx_c = idx.cropped(Vec3D(1,2,3)) # cropping
   >>> idx_c
   VolumetricIndex(resolution=Vec3D(4, 4, 30), bbox=BBox3D(bounds=((4.0, 16.0), (8.0, 20.0), (90.0, 240.0)), unit='nm', pprint_px_resolution=(1, 1, 1)), chunk_id=0, allow_slice_rounding=False)
   >>> print(idx_c.pformat())
   (1.0, 2.0, 3.0) - (4.0, 5.0, 8.0)
   >>> idx_c.shape
   Vec3D(3, 3, 5)
   >>> idx_p = idx.padded(Vec3D(1,2,3)) # padding
   >>> idx_p
   VolumetricIndex(resolution=Vec3D(4, 4, 30), bbox=BBox3D(bounds=((-4.0, 24.0), (-8.0, 36.0), (-90.0, 420.0)), unit='nm', pprint_px_resolution=(1, 1, 1)), chunk_id=0, allow_slice_rounding=False)
   >>> print(idx_p.pformat())
   (-1.0, -2.0, -3.0) - (6.0, 9.0, 14.0)
   >>> idx_p.shape
   Vec3D(7, 11, 17)


Throughout ``zetta_utils``, the ``VolumetricIndex`` is the main way to specify ROIs.

Using ``VolumetricIndex``, the first example above becomes:

.. doctest::

   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> from zetta_utils.layer.volumetric import VolumetricIndex
   >>> from zetta_utils.geometry import Vec3D
   >>> idx = VolumetricIndex.from_coords(
   ...     start_coord = Vec3D(13000, 4000, 2000),
   ...     end_coord = Vec3D(13100, 4100, 2001),
   ...     resolution = Vec3D(64, 64, 40)
   ... )
   >>> cvl = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   ... )
   >>> data = cvl[idx]
   >>> data.shape # channel, x, y, z
   (1, 100, 100, 1)

.. note::
   Since ``VolumetricIndex`` already contains the resolution information, the ``index_resolution`` provided at the initialisation of ``VolumetricLayer`` is overridden when indexing into it using a ``VolumetricIndex``.

   The other two parameters --- ``default_desired_resolution`` and ``data_resolution`` -- function as expected.

Writing to VolumetricLayers
---------------------------

To write to a ``VolumetricLayer``, we need one where we have write access. While FAFB v15 is public-read, it is not public-write. For the walkthrough, we will make a local ``VolumetricLayer``.

.. note::

   When using CloudVolume backends, the credentials for accessing remote volumes are managed in JSON secrets in ``~/.cloudvolume/secrets/``; when using TensorStore backends, the credentials are managed by ``gcloud auth``. Please consult the documentations for either package for details.

Precomputed volumes require an *infofile* that contains information about things like:

* number of channels
* data type
* chunk size in voxels (for each mip)
* chunk offset in voxels (for each mip).

.. collapse:: infofiles in zetta_utils

   In ``zetta_utils``, infofiles are handled by ``zetta_utils.layer.volumetric.precomputed`` module, which is used by ``zetta_utils.layer.volumetric.cloudvol`` and ``zetta_utils.layer.volumetric.tensorstore`` (both instances of ``VolumetricBackend``). While changing the contents of the infofiles within Python (rather than passing in arguments into `build_cv_layer`) is outside the scope of this guide and is something that you shouldn't need to do, here is the example code for reading the content (with ``cvl`` as before):

     >>> cvl.backend.get_bounds(Vec3D(4, 4, 40)) # get bound at resolution
     VolumetricIndex(resolution=Vec3D(4, 4, 40), bbox=BBox3D(bounds=((0, 1048576), (0, 524288), (0, 282560)), unit='nm', pprint_px_resolution=(1, 1, 1)), chunk_id=0, allow_slice_rounding=False)
     >>> cvl.backend.get_chunk_size(Vec3D(4, 4, 40)) # get chunk size at resolution
     Vec3D(1024, 1024, 1)
     >>> cvl.backend.get_voxel_offset(Vec3D(4, 4, 40)) # get voxel offset at resolution
     Vec3D(0, 0, 0)
     >>> cvl.backend.get_dataset_size(Vec3D(4, 4, 40)) # get voxel offset at resolution
     Vec3D(262144, 131072, 7064)


For most common use cases, it will suffice to use an existing infofile as a template. We will do that here, but change the chunk size for (8, 8, 40) nm resolution to be (128, 128, 1) voxels:

.. doctest::

   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> cvl = build_cv_layer(
   ...    path="file://~/zetta_utils_temp/temp",  # path for the volume
   ...    info_reference_path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm", # path for the reference infofile
   ...    info_chunk_size_map={"8_8_40": (128, 128, 1)} # override chunk size - key has to be in "x_y_z" format for CloudVolume
   ... )

Let's try writing to a chunk:

.. doctest::

   >>> from torch import ones, float32
   >>> from zetta_utils.layer.volumetric import VolumetricIndex
   >>> from zetta_utils.geometry import Vec3D
   >>> idx = VolumetricIndex.from_coords(
   ...    start_coord = Vec3D(0, 0, 0),
   ...    end_coord = Vec3D(128, 128, 1),
   ...    resolution = Vec3D(8, 8, 40)
   ... )
   >>> tensor = ones((1, 128, 128, 1), dtype=float32) # requires CXYZ
   >>> cvl[idx] = tensor

Processors in VolumetricLayers
------------------------------

We have covered most of the basic usage of ``VolumetricLayer``, but there is one final aspect that we have to cover: **Processors**. ``Processors`` are ``Callable`` s that, upon a read from or write to a ``VolumetricLayer``, modify the requested index and / or the data. There are three different types of ``Processors``:

#. ``DataProcessor``, which just modifies the data
#. ``IndexProcessor``, which just modifies the index
#. ``JointIndexDataProcessor``, which modifies both the index and the data.

These ``Processors`` imbue ``VolumetricLayer`` with a lot of built-in flexibility. For instance, suppose that we wanted to threshold the normalised FAFB v15 images so that any location with value below 0 was set to 0. Instead of writing code to handle this inside our task, we can simply define a ``DataProcessor`` (which is a protocol) as follows:

.. doctest::

   >>> class ThresholdProcessor:
   ...     def __init__(self, threshold=0):
   ...         self.threshold = threshold
   ...
   ...     def __call__(self, data):
   ...         data[data < self.threshold] = self.threshold
   ...         return data

We initialise the ``VolumetricLayer`` with this ``DataProcessor``, and compare the output to one without:

.. doctest::

   >>> from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
   >>> cvl_without_proc = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm",
   ... )
   >>> cvl_with_proc = build_cv_layer(
   ...    path="https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm",
   ...    read_procs=[ThresholdProcessor(0)]
   ... )
   >>> idx = VolumetricIndex.from_coords(
   ...     start_coord = Vec3D(13000, 4000, 2000),
   ...     end_coord = Vec3D(13100, 4100, 2001),
   ...     resolution = Vec3D(64, 64, 40)
   ... )
   >>> cvl_without_proc[idx].min()
   -2.9491823
   >>> cvl_with_proc[idx].min()
   0.0

This ``VolumetricLayer`` will now apply the ``__call__`` from the ``ThresholdProcessor`` before returning the output for each read.

``read_procs`` is a list, and you can chain arbitrary many ``Processor`` as needed. ``write_procs`` and ``index_procs`` are similar: ``write_procs`` modifies the data to be written before writing, and ``index_procs`` will modify the index to read or write.

The use cases for ``Processors`` include:

* data augmentation for training
* thresholding for masking
* translating the index uniformly

and so much more.

.. collapse:: JointIndexDataProcessor

   The **JointIndexDataProcessor** allows for complex changes to both the index and the data; for instance, consider the rotation augmentation where, given some angle, you wish to download a larger area than the ``VolumetricIndex`` requested (centred at the midpoint of the index), rotate it by the angle, and then crop it to the originally requested size without having any padding in the output. ``JointIndexDataProcessor`` exists to handle such cases, but have a few intricacies:

   * There are separate ``read`` and ``write`` modes that need to be implemented in the protocol
   * When used in ``read_procs``, the order in which the index is processed is reversed (last in the list gets called first), but not the data.

   This design is intended to allow users to use the same list for ``read_procs`` and ``write_procs``.


Builder and CUE files
=====================

Builder
-------

The **builder** provides machinery to represent ``VolumetricLayer``, ``DataProcessor``, ``VolumetricIndex``, or any other registered component as a dictionary. This is used to pass in flexible parameters to CLI tools and to allow readable specifications for workflows through CUE, as we will see in the next subsection.

The registration is done through a decorator at the time of declaration. For instance, we may register the ``ThresholdProcessor`` above like so:

.. doctest::

   >>> from zetta_utils import builder
   >>> @builder.register("ThresholdProcessor")
   ... class ThresholdProcessor:
   ...     def __init__(self, threshold=-1):
   ...         self.threshold = threshold
   ...
   ...     def __call__(self, data):
   ...         data[data < self.threshold] = self.threshold
   ...         return data

After a class has been registered, you can represent an object of that class as a dictionary (called a **spec**) by including the matching ``@type`` key and providing the initialisation parameters:

.. doctest::

   >>> spec = {
   ...     "@type": "ThresholdProcessor",
   ...     "threshold": 10
   ... }
   >>> obj = builder.build(spec)
   >>> print(type(obj))
   <class 'ThresholdProcessor'>
   >>> print(obj.threshold)
   10

The builder can also register methods and functions:

.. doctest::

   >>> @builder.register("echo")
   ... def echo(x):
   ...     return x
   >>> spec = {
   ...     "@type": "echo",
   ...     "x": "some_value"
   ... }
   >>> obj = builder.build(spec)
   >>> print(obj)
   some_value

All user-facing ``zetta_utils`` classes (with one exception) and some other useful classes / methods / functions are registered on module import, and the state of the current registry (i.e., all classes that can be built from the spec within the current session) can be checked out by inspecting ``builder.REGISTRY``.

.. note::

   The exception mentioned above is ``Vec3D``. Because it is used so often and writing ``"@type": Vec3D`` is unwieldy, every registered class accepts a ``Sequence`` of floats or ints and lets the constructor cast to ``Vec3D``.

The ``builder`` will build your objects recursively, which means you can specify complex strucures. For instance, a ``VolumetricLayer`` that has both read and write procs might look like:

.. doctest::

   >>> spec = {
   ...     "@type": "build_cv_layer",
   ...     "path": "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm",
   ...     "read_procs": [
   ...          {
   ...              "@type": "ThresholdProcessor",
   ...              "threshold": 0
   ...          }
   ...     ],
   ...     "write_procs": [
   ...          {
   ...              "@type": "ThresholdProcessor",
   ...              "threshold": 10
   ...          }
   ...     ]
   ... }
   >>> cvl = builder.build(spec)



CUE Files
---------

With the builder, a dictionary is all we need to specify a function call. The dictionary can be specified in any structured language, CUE

``zetta_utils`` uses **CUE** files for configuring a run. `CUE <https://cuelang.org/>`_ is an open-source data validation language that is a superset of JSON.  To proceed with this tutorial, be sure you have cuelang `installed <https://cuelang.org/docs/install/>`_.

.. collapse:: Why not just use Python or JSON?

   CUE has a number of advantages over either Python or JSON for specifying complex tasks:

   * vs. **Python**:
      #. Using CUE separates out code from runtime configuration: it is organisationally clear what contains the configuration versus the actual operations being run.
      #. Configuring a run is less of a mental block, as it does not require coding.
      #. Debugging and maintaining code quality is easier.
      #. Limits what the user can do, which is better for security and readabilty.
   * vs. **JSON**:
      #. CUE allows (simple) logic and variables, which is helpful when specifying a complex task.
      #. CUE can be typed.
   * vs. **Both**:
      #. CUE is more parsimonious and readable.
      #. CUE is associative and commutative: the variables that need to change from run to run can be grouped together (for instance, at the top of the file or under a heading) without affecting the function signature or writing a specific parser, and they can be anywhere in the file.

In fact, the main way that you as the user will be interacting with ``zetta_utils`` is not through a Python shell, but through editing the CUE specs and running them in the CLI. We saw in the previous subsection that functions can be registered with the builder. Running a ``zetta_utils`` command through the CLI is simply the process of asking the builder to run a function with a rather complicated signature specified in the spec, and exit the Python shell.

By the way of comparison, here is an identical spec in JSON and in CUE:

JSON:


.. code-block:: python

  {
      "@type": "LayerDataset",
      "layer": {
          "@type": "build_layer_set",
          "layers": {
              "img": {
                  "@type": "build_cv_layer",
                  "path": "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
              },
              "img_norm": {
                  "@type": "build_cv_layer",
                  "path": "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
              }
          }
      },
      "sample_indexer": {
          "@type": "VolumetricStridedIndexer",
          "bbox": {
             "@type": "BBox3D.from_coords",
             "start_coord": (1000, 1000, 2000),
             "end_coord": (2000, 2000, 2100),
             "resolution": (64, 64, 40),
          },
          "resolution": (64, 64, 40),
          "chunk_size": (128, 128, 1),
          "stride": (32, 32, 1),
          "mode": "shrink",
      }
   }

CUE:

.. code-block:: cue

   "@type": "LayerDataset",
   layer: {
       "@type": "build_layer_set",
       layers: {
           img: {
               "@type": "build_cv_layer",
               path: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
           },
           img_norm: {
               "@type": "build_cv_layer",
               path: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
           }
       }
   },
   sample_indexer: {
       "@type": "VolumetricStridedIndexer",
       bbox: {
          "@type": "BBox3D.from_coords",
          start_coord: [1000, 1000, 2000],
          end_coord: [2000, 2000, 2100],
          resolution: [64, 64, 40],
       },
       resolution: [64, 64, 40],
       chunk_size: [128, 128, 1],
       stride: [32, 32, 1],
       mode: "shrink"
   }


Variables in CUE
----------------

Variables in CUE start with a hashtag. The spec above can be refactored as:

.. code-block:: cue

   #PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
   #PATH_NORM: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   #BBOX: {
             "@type": "BBox3D.from_coords",
             start_coord: [1000, 1000, 2000],
             end_coord: [2000, 2000, 2100],
             resolution: [64, 64, 40],
   }

   "@type": "LayerDataset",
   layer: {
       "@type": "build_layer_set",
       layers: {
           img: {
               "@type": "build_cv_layer",
               path: #PATH
           },
           img_norm: {
               "@type": "build_cv_layer",
               path: #PATH_NORM
           }
       }
   },
   sample_indexer: {
       "@type": "VolumetricStridedIndexer",
       bbox: #BBOX,
       resolution: [64, 64, 40],
       chunk_size: [128, 128, 1],
       stride: [32, 32, 1],
       mode: "shrink"
   }

As noted above, CUE allows you to use variables and declare them later. Furthermore, you can partially declare variables with a placeholder and instantiate them elsewhere, like so:


.. code-block:: cue

   #BBOX_TMPL: {
             "@type": "BBox3D.from_coords",
             start_coord: [1000, 1000, 2000],
             end_coord: [2000, 2000, 2100],
             resolution: _,
   }

   #BBOX: #BBOX_TMPL & {
             resolution: [64, 64, 40]
   }


SubchunkableApplyFlow
=====================

Introduction
------------

**SubchunkableApplyFlow** is the main way that the end users are expected to run inference with ``zetta_utils``. Given an arbitrary chunkwise function or an operation, ``SubchunkableApplyFlow`` provides two key functionalities:

#. The ability to recursively split the provided bounding box into chunks, subchunks, subsubchunks, and so forth, with global parallelisation at the chunk level. (Local parallelisation, which happens at the smallest level, is handled by ``mazepa``.)
#. The ability to specify (subject to divisibility requirements discussed below) arbitrary number of pixels to blend (linear or quadratic) or crop in each dimension at each level.

Chunking with cropping and blending is an absolute necessity for running inference or any other volumetric task in the context of connectomics: because a dataset can be at petascale or even larger, there is no hope of running anything without splitting the dataset into chunks. To mitigate the edge artifacts from chunkwise processing, we can use either cropping or blending. Cropping refers to padding the area to be processed and only writing in the middle of the area; blending refers to padding the areas to be processed and writing out a weighted sum of the outputs from different chunks in the area that overlaps.

One might ask why subchunking is necessary over simple chunking. After all, don't we just need the processing chunk to fit in memory? The short answer is that using chunk-based backends necessitates it: because we have to read and write in chunks, subchunking results in *huge* performance increases over naive chunking. For more details, see the architecture discussion in the main :doc:`SubchunkableApplyFlow documentation <subchunkable_apply_flow>`.

``SubchunkableApplyFlow`` has many arguments (please refer to its docstring for a comprehensive list and usage) but here is an annotated minimal example that simply copies a (4096, 4096, 10) ROI of VolumetricLayer in (1024, 1024, 1) chunks, with no cropping or blending:

.. code-block:: cue

   //
   // Handy variables.
   #SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   #DST_PATH: "file://~/zetta_utils_temp/"
   #BBOX: {
      "@type": "BBox3D.from_coords"
      start_coord: [29696, 16384, 2000]
      end_coord: [29696 + 1024, 16384 + 1024, 2000 + 10]
      resolution: [16, 16, 40]
   }

   // We are asking the builder to call mazepa.execute with the following target
   "@type": "mazepa.execute"
   target: {
      // We're applying subchunkable processing flow.
      "@type": "build_subchunkable_apply_flow"

      // This is the bounding box for the run
      bbox: #BBOX

      // What resolution is our destination?
      dst_resolution: [16, 16, 40]

      // How do we chunk/crop/blend? List of lists for subchunking.
      processing_chunk_sizes: [[1024, 1024, 1]]
      processing_crop_pads: [[0, 0, 0]]
      processing_blend_pads: [[0, 0, 0]]

      // Flag to indicate "simple" processing mode where outputs get
      // written directly to the destination layer. Don't worry about
      // this for now.
      skip_intermediaries: true

      // Specification for the operation we're performing.
      fn: {
         "@type":    "lambda"
         lambda_str: "lambda src: src"
      }
      // Specification for the inputs to the operation;
      // Our lambda expects a single kwarg called 'src'.
      op_kwargs: {
         src: {
            "@type": "build_cv_layer"
            path:    #SRC_PATH
         }
      }

      // Specification of the output layer. Subchunkable expects
      // a single output layer. If multiple output layers are
      // needed, refer to advanced examples.
      dst: {
         "@type":             "build_cv_layer"
         path:                #DST_PATH
         info_reference_path: #SRC_PATH
         on_info_exists:      "overwrite"
      }
   }

To run this CUE file, you can copy the code block to ``example.cue`` and then do ``zetta run example.cue``.

Note the arguments ``processing_chunk_sizes``, ``processing_crop_pads``, ``processing_blend_pads``; these are list of lists, going from the highest (largest) to the lowest (smallest) level of subchunks. In this example, there is only a single level, so these arguments have length one.

Let's say you wanted to pad each 1024x1024x1 input chunk by 256 pixels in each direction in XY for processing. In ``SubchunkableApplyFlow``, the processing chunks are always specified in the desired output size, so cropping is represented as ``crop_pad``. This means that the chunk will be padded by the ``crop_pad`` amount in each direction, processed, and then cropped to return the ``processing_chunk_size`` specified.

.. code-block:: cue

      // Take each chunk of 1024x1024x1, pad to 1536x1536x1, process, and return 1024x1024x1
      processing_chunk_sizes: [[1024, 1024, 1]]
      processing_crop_pads: [[256, 256, 0]]

What if you wanted to use blending? Blending is also specified as a padding: given the (1024, 1024, 1) ``processing_chunk_size``, specifying a ``blend_pad`` of 256 pixels is equivalent to specifying 512 pixel overlap between each (1536, 1536, 1) chunk. You can specify ``blend_mode`` of either ``linear`` or ``quadratic`` (default). However, if you wish to use blending for any given level, you **must** specify a location for temporary intermediary layers.

.. code-block:: cue

      // Take each chunk of 1024x1024x1, pad to 1536x1536x1, process, and blend.
      processing_chunk_sizes: [[1024, 1024, 1]]
      processing_blend_pads: [[256, 256, 0]]
      processing_blend_modes: ["linear"]

      // Where to put the temporary layers.
      level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/"]

.. note::

   You may use arbitrary ``crop_pad``, but ``blend_pad`` **must** be at most one half of the ``processing_chunk_size`` for that level.


Subchunking
-----------

Let's take the above example and modify it slightly to use subchunking, so that each (1024, 1024, 1) chunk is split into (256, 256, 1) subchunks. Two things need to be changed:

#. The three ``processing_`` arguments need to be lengthened to length 2.
#. The smallest subchunk size (256) is smaller than the backend chunk size of the destination layer (1024). Because of this, we must remove the ``skip_intermediaries: true`` and instead include ``level_intermediaries_dirs``. The immediate operation results will be first written to the intermediary location, and then later copied over to the final destination layer with the correct chunk size.

.. collapse:: Intermediary Directories

   The ``level_intermediaries_dirs`` must be specified whenever ``skip_intermediaries``
   is not used. ``skip_intermediaries`` cannot be used when non-zero ``blend_pad`` or
   ``roi_crop_pad`` is used.

With the changes, the example above becomes:

.. code-block:: cue
  :emphasize-lines: 23, 24, 25, 27, 28

   #SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   #DST_PATH: "file://~/zetta_utils_temp/"
   #BBOX: {
      "@type": "BBox3D.from_coords"
      start_coord: [29696, 16384, 2000]
      end_coord: [29696 + 1024, 16384 + 1024, 2000 + 10]
      resolution: [16, 16, 40]
   }

   // We are asking the builder to call mazepa.execute with the following target.
   "@type": "mazepa.execute"
   target: {
      // We're applying subchunkable processing flow.
      "@type": "build_subchunkable_apply_flow"

      // This is the bounding box for the run
      bbox: #BBOX

      // What resolution is our destination?
      dst_resolution: [16, 16, 40]

      // How do we chunk/crop/blend? List of lists for subchunking.
      processing_chunk_sizes: [[1024, 1024, 1], [256, 256, 1]]
      processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
      processing_blend_pads: [[0, 0, 0], [0, 0, 0]]

      // Where to put the temporary layers.
      level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/", "file://~/.zetta_utils/tmp/"]

      // Specification for the operation we're performing.
      fn: {
         "@type":    "lambda"
         lambda_str: "lambda src: src"
      }
      // Specification for the inputs to the operation;
      // Our lambda expects a single kwarg called 'src'.
      op_kwargs: {
         src: {
            "@type": "build_cv_layer"
            path:    #SRC_PATH
         }
      }

      // Specification of the output layer. Subchunkable expects
      // a single output layer. If multiple output layers are
      // needed, refer to advanced examples.
      dst: {
         "@type":             "build_cv_layer"
         path:                #DST_PATH
         info_reference_path: #SRC_PATH
         on_info_exists:      "overwrite"
      }
   }

Each level can have its own crop and blend (as well as ``blend_mode``), but there are two caveats:

#. **For each level, the processing chunk must evenly divide the ``crop`` and ``blend`` padded processing chunk of the level above**. This is because ``SubchunkableApplyFlow`` uses a recursive implementation, where each padded processing chunk is split into smaller processing subchunks.
#. If you are using blend for any level, you must specify ``max_reduction_chunk_sizes``, which specifies the maximum size of the reduction chunk. When blending is specified, the overlapping, padded outputs are written to separate layers within the intermediary directory, before they are combined (reduced) into the final output based on the `processing_blend_modes` for that level. The reduction operation is also chunked, and ``SubchunkableApplyFlow`` automatically handles the combining of multiple processing chunks into backend-aligned reduction chunks, up to the ``max_reduction_chunk_size`` specified. ``max_reduction_chunk_sizes`` can be given as a single list for all levels or as a list of lists like ``processing_blend/crop_pads``, but it is recommended to set it as large as possible because I/O operations are more efficient with larger chunks.

.. code-block:: cue
  :emphasize-lines: 24, 25, 26, 27, 28, 29, 30, 31, 33, 34

   //
   // Handy variables.
   #SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   #DST_PATH: "file://~/zetta_utils_temp/"
   #BBOX: {
      "@type": "BBox3D.from_coords"
      start_coord: [29696, 16384, 2000]
      end_coord: [29696 + 1024, 16384 + 1024, 2000 + 10]
      resolution: [16, 16, 40]
   }

   // We are asking the builder to call mazepa.execute with the following target.
   "@type": "mazepa.execute"
   target: {
      // We're applying subchunkable processing flow.
      "@type": "build_subchunkable_apply_flow"

      // This is the bounding box for the run
      bbox: #BBOX

      // What resolution is our destination?
      dst_resolution: [16, 16, 40]

      // How do we chunk/crop/blend? List of lists for subchunking.
      // Note that 1024 + 96 * 2 + 32 * 2 = 1280 is evenly divisible by 256.
      // The bottom level can have whatever crop_pad and blend_pad set
      // without divisibility considerations.
      processing_chunk_sizes: [[1024, 1024, 1], [256, 256, 1]]
      processing_crop_pads: [[96, 96, 0], [24, 24, 0]]
      processing_blend_pads: [[32, 32, 0], [16, 16, 0]]
      processing_blend_modes: ["linear", "quadratic"]

      // How large can our reduction chunks be?
      max_reduction_chunk_sizes: [2048, 2048, 1]

      // Where to put the temporary layers.
      level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/", "file://~/.zetta_utils/tmp/"]

      // Specification for the operation we're performing.
      fn: {
         "@type":    "lambda"
         lambda_str: "lambda src: src"
      }
      // Specification for the inputs to the operation;
      // Our lambda expects a single kwarg called 'src'.
      op_kwargs: {
         src: {
            "@type": "build_cv_layer"
            path:    #SRC_PATH
         }
      }

      // Specification of the output layer. Subchunkable expects
      // a single output layer. If multiple output layers are
      // needed, refer to advanced examples.
      dst: {
         "@type":             "build_cv_layer"
         path:                #DST_PATH
         info_reference_path: #SRC_PATH
         on_info_exists:      "overwrite"
      }
   }

Running Remotely
----------------

This subsection assumes that you have followed the GCS and SQS part of the *Getting Started* section of the :doc:`main documentation <index>`.

Once you have a valid spec, getting ``SubchunkableApplyFlow`` to run on a remote cluster on Google Cloud Platform using SQS queues is very easy:

#. Select your project in the GCP Console, and make a new cluster in the Kubernetes engine.
#. Build and push your docker image using ``python build_image.py --project {PROJECT} --tag {tag}`` [`build_image.py <https://github.com/ZettaAI/zetta_utils/blob/main/build_image.py>`_]
#. Modify the CUE file to use remote execution.
#. Run ``zetta run file.cue``.

To modify the CUE file, we change ``mazepa.execute`` to ``mazepa.execute_on_gcp_with_sqs``, and add the required parameters specifying which Kubernetes cluster to use, with what resources and docker image, and how many workers:

.. code-block:: cue
  :emphasize-lines: 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23

   //
   // Handy variables.
   #SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img_norm"
   #DST_PATH: "file://~/zetta_utils_temp/"
   #BBOX: {
      "@type": "BBox3D.from_coords"
      start_coord: [29696, 16384, 2000]
      end_coord: [29696 + 1024, 16384 + 1024, 2000 + 10]
      resolution: [16, 16, 40]
   }

   // Execution parameters
   "@type":                "mazepa.execute_on_gcp_with_sqs"
   worker_image:           "us.gcr.io/{PROJECT}/zetta_utils:{tag}"
   worker_cluster_name:    "{CLUSTER_NAME}" // Kubernetes cluster
   worker_cluster_region:  "us-east1"       // Kubernetes cluster region
   worker_cluster_project: "zetta-research" // Project that the Kubernetes cluster belongs to
   worker_resources: {
      memory: "18560Mi"       // Memory required for each instance
      //"nvidia.com/gpu": "1" // Uncomment if GPU is needed
   }
   worker_replicas: 10   // Number of workers
   local_test:      true // set to `false` execute remotely
   target: {
      // We're applying subchunkable processing flow.
      "@type": "build_subchunkable_apply_flow"

      // This is the bounding box for the run
      bbox: #BBOX

      // What resolution is our destination?
      dst_resolution: [16, 16, 40]

      // How do we chunk/crop/blend? List of lists for subchunking.
      // Note that 1024 + 96 * 2 + 32 * 2 = 1280 is evenly divisible by 256.
      // The bottom level can have whatever crop_pad and blend_pad set
      // without divisibility considerations.
      processing_chunk_sizes: [[1024, 1024, 1], [256, 256, 1]]

      // Where to put the temporary layers.
      level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/", "file://~/.zetta_utils/tmp/"]

      // Specification for the operation we're performing.
      fn: {
         "@type":    "lambda"
         lambda_str: "lambda src: src"
      }
      // Specification for the inputs to the operation;
      // Our lambda expects a single kwarg called 'src'.
      op_kwargs: {
         src: {
            "@type": "build_cv_layer"
            path:    #SRC_PATH
         }
      }

      // Specification of the output layer. Subchunkable expects
      // a single output layer. If multiple output layers are
      // needed, refer to advanced examples.
      dst: {
         "@type":             "build_cv_layer"
         path:                #DST_PATH
         info_reference_path: #SRC_PATH
         on_info_exists:      "overwrite"
      }
   }

When you run this file (with ``local_test`` set to ``false``), ``zetta_utils`` will automatically take care of setting up SQS queues with a MUID (Memorable Unique Identifier), as well as creating a deployment within the Kubernetes cluster with the requested resource.

.. note::

  When cancelling a run in progress, do **NOT** press *Ctrl-C* multiple times. If you press *Ctrl-C* once, ``zetta_utils`` will prompt for confirmation of the cancellation, and gracefully garbage collect the SQS queues and the deployment before returning.

.. warning::
  **If the run force quits and the garbage collector has not been configured, the deployment may run indefinitely!**
