"""Convnet architecture subpackage — lazily resolved via zetta_utils.common.lazy."""
from typing import TYPE_CHECKING

from zetta_utils.common.lazy import make_lazy_module

_LAZY_SUBPACKAGES = ("primitives", "deprecated")

_LAZY_REEXPORTS = {
    ".convblock": ("ConvBlock",),
    ".unet": ("UNet",),
}

__getattr__, __dir__ = make_lazy_module(
    __name__, globals(), _LAZY_SUBPACKAGES, _LAZY_REEXPORTS
)

if TYPE_CHECKING:
    from . import deprecated, primitives
    from .convblock import ConvBlock
    from .unet import UNet
