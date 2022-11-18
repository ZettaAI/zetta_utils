from __future__ import annotations

from typing import Any, Optional, Sequence, TypeVar, Union

import torch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.layer import IndexChunker, Layer
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.tensor_typing import Tensor

from . import build_chunked_volumetric_flow_type

IndexT = TypeVar("IndexT", bound=VolumetricIndex)


def _interpolate(
    src_data: torch.Tensor,
    scale_factor: Union[float, Sequence[float]],
    mode: tensor_ops.InterpolationMode,
    mask_value_thr: float = 0,
) -> torch.Tensor:
    return tensor_ops.interpolate(
        data=src_data,
        scale_factor=scale_factor,
        mode=mode,
        mask_value_thr=mask_value_thr,
        unsqueeze_input_to=5,
    )


@builder.register("chunked_interpolate_xy")
def chunked_interpolate_xy(
    chunker: IndexChunker[IndexT],
    idx: IndexT,
    src: Layer[Any, IndexT, Tensor],
    scale_factor: float,
    mode: tensor_ops.InterpolationMode,
    dst: Optional[Layer[Any, IndexT, Tensor]] = None,
    mask_value_thr: float = 0,
) -> mazepa.Flow:
    if dst is None:
        dst = src
    scale_factor_tuple = [scale_factor, scale_factor, 1]
    dst_idx_res = [
        idx.resolution[0] / scale_factor,
        idx.resolution[1] / scale_factor,
        idx.resolution[2],
    ]
    flow_type = build_chunked_volumetric_flow_type(
        fn=_interpolate,
        chunker=chunker,
        dst_idx_res=dst_idx_res,
    )
    result = flow_type(
        idx=idx,
        dst=dst,
        src=src,
        scale_factor=scale_factor_tuple,
        mode=mode,
        mask_value_thr=mask_value_thr,
    )
    return result
