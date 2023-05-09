# pylint: disable=unused-import, import-outside-toplevel
"""Zetta AI Computational Connectomics Toolkit."""

from . import log, typing, parsing, builder, common
from . import geometry, distributions, layer, ng

log.add_supress_traceback_module(builder)


def load_all_modules():
    load_inference_modules()
    load_training_modules()
    from . import mazepa_addons
    from . import cloud


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


def load_inference_modules():
    from . import (
        alignment,
        augmentations,
        convnet,
        mazepa,
        mazepa_layer_processing,
        segmentation,
        tensor_ops,
        tensor_typing,
        tensor_mapping,
    )
    from .layer import volumetric
    from .layer.volumetric import cloudvol


def load_training_modules():
    from . import (
        alignment,
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
