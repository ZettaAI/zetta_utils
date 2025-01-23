import torch

from zetta_utils.tensor_ops import convert

from .backend import VolumetricBackend
from .build import build_volumetric_layer
from .constant import ConstantVolumetricBackend, build_constant_volumetric_layer
from .frontend import (
    SliceUserVolumetricIndex,
    UnconvertedUserVolumetricIndex,
    UserVolumetricIndex,
    VolumetricFrontend,
)
from .index import VolumetricIndex
from .layer import VolumetricLayer
from .layer_set import VolumetricLayerSet, build_volumetric_layer_set
from .protocols import VolumetricBasedLayerProtocol
from .tensorstore import build
from .tools import (
    DataResolutionInterpolator,
    InvertProcessor,
    ROIMaskProcessor,
    VolumetricIndexChunker,
    VolumetricIndexOverrider,
    VolumetricIndexPadder,
    VolumetricIndexScaler,
    VolumetricIndexTranslator,
)

VolumetricLayerDType = torch.Tensor
to_vol_layer_dtype = convert.to_torch
