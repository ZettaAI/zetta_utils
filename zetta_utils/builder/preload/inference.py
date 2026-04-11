# pylint: disable=unused-import, wrong-import-position
"""Inference module imports."""

import time

from zetta_utils import log

_start = time.perf_counter()

# Import core first
from zetta_utils.builder.preload import core

from zetta_utils import (
    augmentations,
    convnet,
    mazepa,
    mazepa_layer_processing,
    tensor_ops,
    tensor_typing,
    tensor_mapping,
)
from zetta_utils.layer import volumetric
from zetta_utils.layer.volumetric import cloudvol
from zetta_utils.message_queues import sqs

from zetta_utils import mazepa_addons
from zetta_utils import message_queues
from zetta_utils import cloud_management

from zetta_utils import internal

_elapsed = time.perf_counter() - _start
log.get_logger("zetta_utils").debug(f"Preload inference modules: {_elapsed:.2f}s")
