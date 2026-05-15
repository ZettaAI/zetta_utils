"""Layer subpackage exports — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("protocols", "layer_set")

_LAZY_REEXPORTS = {
    ".backend_base": ("Backend",),
    ".tools_base": (
        "JointIndexDataProcessor",
        "IndexChunker",
        "DataProcessor",
        "IndexProcessor",
    ),
    ".layer_base": ("Layer",),
    ".layer_set": ("build_layer_set",),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import layer_set, protocols
    from .backend_base import Backend
    from .layer_base import Layer
    from .layer_set import build_layer_set
    from .tools_base import DataProcessor, IndexChunker, IndexProcessor, JointIndexDataProcessor
