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


@builder.register("seg_to_rgb")
@typechecked
def seg_to_rgb(
    data: TensorTypeVar,
) -> TensorTypeVar:
    """
    Transform a segmentation into an RGB map.

    :param data: Input segmentation
    """
    assert 2 <= data.ndim <= 5
    data_np = convert.to_np(data)
    data_np = data_np[0, ...] if data_np.ndim > 4 else data_np
    data_np = data_np[0, ...] if data_np.ndim > 3 else data_np
    unq, unq_inv = np.unique(data_np, return_inverse=True)

    # pylint: disable=invalid-name
    # Random colormap
    N = len(unq)
    R = np.random.rand(N)
    G = np.random.rand(N)
    B = np.random.rand(N)

    # Background
    idx = unq == 0
    R[idx] = G[idx] = B[idx] = 0

    R = R[unq_inv].reshape(data_np.shape)
    G = G[unq_inv].reshape(data_np.shape)
    B = B[unq_inv].reshape(data_np.shape)
    rgbmap = np.stack((R, G, B), axis=0).astype(np.float32)
    result = convert.astype(rgbmap, data)
    return result
