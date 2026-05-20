"""Tensor ops subpackage exports — lazily resolved via zetta_utils.common.lazy.

The pre-lazy version had to import ``common``/``convert``/``label``/``mask``
before ``generators`` to dodge a circular import (``common.py`` does
``from zetta_utils import tensor_ops`` for runtime use of
``tensor_ops.convert.*``). Lazy resolution sidesteps the cycle entirely:
every submodule loads only on first access, by which point the package is
fully initialized.
"""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = (
    "common",
    "convert",
    "filtering",
    "generators",
    "label",
    "mask",
    "multitensor",
    "normalization",
    "projection",
    "traceback_supress",
)

_LAZY_REEXPORTS = {
    ".common": (
        "InterpolationMode",
        "compare",
        "crop",
        "crop_center",
        "interpolate",
        "squeeze",
        "unsqueeze",
        "unsqueeze_to",
        "pad_center_to",
    ),
    ".convert": ("astype", "to_np", "to_torch"),
    ".label": ("get_disp_pair", "seg_to_aff", "seg_to_rgb"),
    ".mask": ("filter_cc",),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import (
        common,
        convert,
        filtering,
        generators,
        label,
        mask,
        multitensor,
        normalization,
        projection,
        traceback_supress,
    )
    from .common import (
        InterpolationMode,
        compare,
        crop,
        crop_center,
        interpolate,
        pad_center_to,
        squeeze,
        unsqueeze,
        unsqueeze_to,
    )
    from .convert import astype, to_np, to_torch
    from .label import get_disp_pair, seg_to_aff, seg_to_rgb
    from .mask import filter_cc
