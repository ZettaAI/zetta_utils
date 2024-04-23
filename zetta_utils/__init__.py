# pylint: disable=unused-import, import-outside-toplevel
"""Zetta AI Computational Connectomics Toolkit."""
from . import log, typing, parsing, builder, common, constants
from . import geometry, distributions, layer, ng

builder.registry.MUTLIPROCESSING_INCOMPATIBLE_CLASSES.add("mazepa")
builder.registry.MUTLIPROCESSING_INCOMPATIBLE_CLASSES.add("lightning")
log.add_supress_traceback_module(builder)


def load_all_modules():
    load_inference_modules()
    load_training_modules()


def try_load_train_inference():  # pragma: no cover
    try:
        load_inference_modules()

    except ImportError:
        ...

    try:
        load_training_modules()
    except ImportError:
        ...

    try:
        from . import mazepa_addons
    except ImportError:
        ...


def try_load_submodules():  # pragma: no cover
    try:
        from . import internal
    except ImportError:
        ...


def load_inference_modules():
    from . import (
        augmentations,
        convnet,
        mazepa,
        mazepa_layer_processing,
        tensor_ops,
        tensor_typing,
        tensor_mapping,
    )
    from .layer import volumetric
    from .layer.volumetric import cloudvol
    from .message_queues import sqs

    from . import mazepa_addons
    from . import message_queues
    from . import cloud_management

    try_load_submodules()


def load_training_modules():
    from . import (
        augmentations,
        convnet,
        mazepa,
        tensor_ops,
        tensor_typing,
        training,
        tensor_mapping,
    )
    from .layer import volumetric
    from .layer.volumetric import cloudvol

    from . import mazepa_addons
    from . import message_queues
    from . import cloud_management

    try_load_submodules()


try_load_train_inference()
