from typing import Literal, Sequence, Union

import numpy as np
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_ops.common import supports_dict
from zetta_utils.tensor_typing import TensorTypeVar


@builder.register("first_hit_projection")
@supports_dict
@typechecked
def first_hit_projection(
    data: TensorTypeVar,
    bg_color: Union[float, Sequence[float]] = 0.0,
    axis: int = 3,
    direction: Literal["first", "last"] = "first",
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> TensorTypeVar:
    """
    Project 4D (CXYZ) to 3D (default CXY1) by taking first or last non-background value.

    :param data:  Input tensor (CXYZ).
    :param bg_color:  scalar or tuple/array of length C for per-channel background
    :param axis:  axis along which to project (default 3 for z-axis)
    :param direction:  'first' (default) or 'last'
    :param rtol: The relative tolerance parameter (see np.isclose).
    :param atol: The absolute tolerance parameter (see np.isclose).
    :return: Projection with singleton dimension on projected axis
    """
    data_np = convert.to_np(data)

    if direction == "last":
        data_np = np.flip(data_np, axis=axis)

    bg_color_broadcast = np.array(bg_color, dtype=data_np.dtype)
    if bg_color_broadcast.ndim == 0:
        bg_color_broadcast = np.full(data_np.shape[0], bg_color_broadcast)
    bg_color_broadcast = bg_color_broadcast.reshape(-1, 1, 1, 1)

    bg_channel_mask = np.isclose(data_np, bg_color_broadcast, rtol=rtol, atol=atol, equal_nan=True)
    valid_voxel_mask = np.any(~bg_channel_mask, axis=0, keepdims=True)

    first_idx = np.argmax(valid_voxel_mask, axis=axis, keepdims=True)
    has_valid = np.any(valid_voxel_mask, axis=axis, keepdims=True)

    projection = np.take_along_axis(data_np, first_idx, axis=axis)
    projection = np.where(has_valid, projection, bg_color_broadcast)

    return convert.astype(projection, reference=data)
