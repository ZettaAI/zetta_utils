"""GCloud APIs — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_REEXPORTS = {
    ".artifact_registry": ("check_image_exists",),
    ".pricing": ("get_compute_sku_groups",),
    ".instance": ("get_node_info",),
    ".gcs": (
        "get_bucket_location",
        "get_bucket_location_info",
        "is_region_compatible",
    ),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), reexports_by_module=_LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from .artifact_registry import check_image_exists
    from .gcs import (
        get_bucket_location,
        get_bucket_location_info,
        is_region_compatible,
    )
    from .instance import get_node_info
    from .pricing import get_compute_sku_groups
