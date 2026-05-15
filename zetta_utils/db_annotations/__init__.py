"""DB annotations subpackage exports — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("utils",)

_LAZY_REEXPORTS = {
    ".annotation": (
        "read_annotation",
        "read_annotations",
        "add_annotation",
        "add_annotations",
        "update_annotation",
        "update_annotations",
        "parse_ng_annotations",
    ),
    ".collection": (
        "read_collection",
        "read_collections",
        "add_collection",
        "update_collection",
    ),
    ".layer_group": (
        "read_layer_group",
        "read_layer_groups",
        "add_layer_group",
        "update_layer_group",
    ),
    ".layer": (
        "read_layer",
        "read_layers",
        "add_layer",
        "update_layer",
    ),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import utils
    from .annotation import (
        add_annotation,
        add_annotations,
        parse_ng_annotations,
        read_annotation,
        read_annotations,
        update_annotation,
        update_annotations,
    )
    from .collection import (
        add_collection,
        read_collection,
        read_collections,
        update_collection,
    )
    from .layer import add_layer, read_layer, read_layers, update_layer
    from .layer_group import (
        add_layer_group,
        read_layer_group,
        read_layer_groups,
        update_layer_group,
    )
