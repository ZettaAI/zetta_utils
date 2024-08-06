import torch

from zetta_utils.tensor_ops import convert
from .index import (
    VolumetricIndex,
)
from .backend import VolumetricBackend
from .frontend import (
    VolumetricFrontend,
    UserVolumetricIndex,
    UnconvertedUserVolumetricIndex,
    SliceUserVolumetricIndex,
)


from .tools import (
    DataResolutionInterpolator,
    InvertProcessor,
    ROIMaskProcessor,
    VolumetricIndexPadder,
    VolumetricIndexTranslator,
    VolumetricIndexChunker,
    VolumetricIndexOverrider,
    VolumetricIndexScaler,
)
from .layer import (
    VolumetricLayer,
)
from .build import build_volumetric_layer

from .constant import ConstantVolumetricBackend, build_constant_volumetric_layer
from .layer_set import VolumetricLayerSet, build_volumetric_layer_set
from .protocols import VolumetricBasedLayerProtocol

VolumetricLayerDType = torch.Tensor
to_vol_layer_dtype = convert.to_torch
