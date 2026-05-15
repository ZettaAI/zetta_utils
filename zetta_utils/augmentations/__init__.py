"""Augmentations subpackage exports — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("tensor", "misalign")

_LAZY_REEXPORTS = {
    ".common": ("prob_aug",),
    ".imgaug": ("imgaug_augment",),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import misalign, tensor
    from .common import prob_aug
    from .imgaug import imgaug_augment
