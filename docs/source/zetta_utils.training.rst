``zetta_utils.training``
========================

``Overview``
------------

``zetta_utils.training`` module provides standardized integrations for neural network training. It is not meant to be a heaviweight integration that dictates the way in which training must be done, but rather is a set independently useful, lightweight components. Some of the advantages of ``zetta_utils.training`` are:
* Bells and whistles on top of Pytorch Lightning (PL). We've parametrized PL to include extensive checkpointing, better progress bars, Weights and Biases (wandb) integration, and so on so that you don't have to.
* Remote taining management becomes easy for ``zetta_utils.training`` users. Unlimited number of concurrent remote traiing runs can be started, monitored, and easily cancelled from the command line. ``zetta_utils`` can also handle more complicated cluster setups needed for DDP (WIP).
* Integration with ``zetta_utils.builder`` provides a standard way to provide full traiing parametrization in a single spec file. Having a standard single-spec parametrization format simplifies collaboration between team members, and also gets automatically uploaded to wandb which simplifies experimetn management.
* It is easy for ``zetta_utils`` users to share data augmentations and architectures with each other.
* Powerful training-inference integration.


All trainings start with a call to  ``zetta_utils.training.lightning.train``. Other than training and validation dataloaders and checkpoint paths, this function is given a training :italic:`regime` and a :italic:`trainer`. Regime defines the specifics of how the given network is to be trained. This includes training loss calulation, validation loss calculation, any actions that need to be taken at the beginning or end of each validation epoch, etc. Regimes are usually created by the scientist performing experiments.
Trainer defines training loop behavior that is commons for all regimes, such as logging, checkpointing, gradient clipping, etc, Zetta engineering team maintains a default trainer configuration that can be accessed through
``zetta_utils.training.lightning.trainers.build_default_trainer``. Trainer extensions are usually developed by the engineering team.

You can find existing reimes in ``zetta_utils/training/lightning/regimes``.
You can find example training specs that use some of those regimes in ``specs/examples/training``. To learn more about the spec file format, refer to ``zetta_utils.builder`` documetnation.

API reference
-------------
``zetta_utils.training.lightning``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: zetta_utils.training.lightning.train


``zetta_utils.training.datasets``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: zetta_utils.training.datasets.LayerDataset

.. autoclass:: zetta_utils.training.datasets.JointDataset

.. autoclass:: zetta_utils.training.datasets.sample_indexers.ChainIndexer

.. autoclass:: zetta_utils.training.datasets.sample_indexers.RandomIndexer

.. autoclass:: zetta_utils.training.datasets.sample_indexers.VolumetricStridedIndexer

.. autofunction:: zetta_utils.training.datasets.sample_indexers.VolumetricStridedIndexer.__call__
