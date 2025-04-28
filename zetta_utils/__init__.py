# pylint: disable=unused-import, import-outside-toplevel, broad-exception-caught, import-error
"""Zetta AI Computational Connectomics Toolkit."""
import os
import sys
import warnings

try:
    import graph_tool
except Exception:  # pragma: no cover
    ...

from . import log, typing, parsing, builder, common, constants
from . import geometry, distributions, layer, ng
from .log import get_logger

if "sphinx" not in sys.modules:
    import pdbp  # noqa

    os.environ["PYTHONBREAKPOINT"] = "pdbp.set_trace"

logger = get_logger("zetta_utils")
ignore_warnings_from = [
    "python_jsonschema_objects",
    "kornia",
    "google",
    "pytorch_lightning",
    "lightning_fabric",
    "pkg_resources",
]

for pkg_name in ignore_warnings_from:
    warnings.filterwarnings("ignore", module=pkg_name)
warnings.filterwarnings("ignore", category=DeprecationWarning)

builder.registry.MULTIPROCESSING_INCOMPATIBLE_CLASSES.add("mazepa")
builder.registry.MULTIPROCESSING_INCOMPATIBLE_CLASSES.add("lightning")
log.add_supress_traceback_module(builder)


def load_all_modules():
    load_inference_modules()
    load_training_modules()
    from . import task_management


def try_load_train_inference():  # pragma: no cover
    try:
        load_inference_modules()

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)

    try:
        load_training_modules()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)

    try:
        from . import mazepa_addons
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)


def load_submodules():  # pragma: no cover
    from . import internal


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

    load_submodules()


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
    from .layer import volumetric, db_layer
    from .layer.db_layer import datastore, firestore
    from .layer.volumetric import cloudvol

    from . import mazepa_addons
    from . import message_queues
    from . import cloud_management

    load_submodules()
