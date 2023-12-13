from __future__ import annotations

from typing import Sequence, overload

import numpy as np
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry import IntVec3D
from zetta_utils.tensor_typing import TensorTypeVar

from . import convert


@builder.register("get_disp_pair")
@typechecked
def get_disp_pair(
    data: TensorTypeVar,
    disp: Sequence[int],
) -> tuple[TensorTypeVar, TensorTypeVar]:
    """
    Get a pair of same-sized volumes displaced with respect to each other by
    the given displacement vector.

    :param data: Input segmentation
    :param disp: Displacement vector
    """
    ndim = len(disp)
    assert ndim == 3
    disp = IntVec3D(*disp)

    assert data.ndim >= ndim
    for a, b in zip(data.shape[-ndim:], np.absolute(disp)):
        assert a > b

    disp1 = np.maximum(disp, 0)
    disp2 = np.maximum(-disp, 0)

    slices1 = [slice(0, None) for _ in range(data.ndim - ndim)]
    slices2 = [slice(0, None) for _ in range(data.ndim - ndim)]
    for shape, offset1, offset2 in zip(data.shape[-ndim:], disp1, disp2):
        slices1.append(slice(offset1, shape - offset2))
        slices2.append(slice(offset2, shape - offset1))

    return data[tuple(slices1)], data[tuple(slices2)]


@overload
def seg_to_aff(
    data: TensorTypeVar,
    edge: Sequence[int],
    mask: TensorTypeVar = ...,
) -> tuple[TensorTypeVar, TensorTypeVar]:
    ...


@overload
def seg_to_aff(
    data: TensorTypeVar,
    edge: Sequence[int],
    mask: None = ...,
) -> TensorTypeVar:
    ...


@builder.register("convert_seg_to_aff")
@typechecked
def seg_to_aff(
    data,
    edge,
    mask=None,
):
    """
    Transform a segmentation into an affinity map characterized by the given
    `edge`

    :param data: Input segmentation 3D volume
    :param edge: Edge, meaning an offset vector
    :param mask: Binary mask for `data`
    """
    pair = get_disp_pair(data, edge)
    aff = (pair[0] == pair[1]) & (pair[0] != 0) & (pair[1] != 0)
    aff = convert.astype(aff, data, cast=True)

    result = aff
    if mask is not None:
        assert data.shape == mask.shape
        pair = get_disp_pair(mask, edge)
        affmsk = convert.astype(pair[0] * pair[1], mask, cast=True)
        result = aff, affmsk
    return result
