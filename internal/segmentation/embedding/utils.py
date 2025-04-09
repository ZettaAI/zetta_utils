from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F
from typeguard import typechecked

from zetta_utils import builder, tensor_ops
from zetta_utils.tensor_ops import convert
from zetta_utils.tensor_typing import TensorTypeVar


@builder.register("vec_to_pca")
@typechecked
def vec_to_pca(
    data: TensorTypeVar,
    pca_crop_pad: Sequence[int] | None = None,
) -> TensorTypeVar:
    """
    Transform feature vectors into an RGB map with PCA dimensionality reduction.

    :param data: Feature vectors
    :param pca_crop_pad: Specify in xyz how much to crop input before running PCA
    :return: PCA'ed feature vectors
    """
    assert (data.ndim == 5) and (data.shape[0] == 1)
    if pca_crop_pad is not None:
        assert len(pca_crop_pad) == 3

    data_np = convert.to_np(data).astype(np.float32)

    # pylint: disable=invalid-name
    dim = data_np.shape[-4]
    data_tp = data_np.transpose(0, 2, 3, 4, 1)

    data_tp_cropped = data_tp
    if pca_crop_pad:
        data_tp_cropped = data_tp[
            :,
            pca_crop_pad[0] : -pca_crop_pad[0],
            pca_crop_pad[1] : -pca_crop_pad[1],
            pca_crop_pad[2] : -pca_crop_pad[2],
            :,
        ]

    X = data_tp.reshape(-1, dim)  # (n_samples, n_features)
    X_cropped = data_tp_cropped.reshape(-1, dim)  # (n_samples, n_features)
    fit = PCA(dim).fit(X_cropped)
    pca = fit.transform(X)
    pca_tp = pca.reshape(data_tp.shape)
    pca_np = pca_tp.transpose(0, 4, 1, 2, 3)
    result = convert.astype(pca_np, data)
    return result


@builder.register("vec_to_rgb")
@typechecked
def vec_to_rgb(
    data: TensorTypeVar,
    normalization_crop_pad: Sequence[int] | None = None,
    per_channel_normalization: bool = False,
) -> TensorTypeVar:
    """
    Transform feature vectors into an RGB map by slicing the first three
    channels and then rescale them.

    :param data: Feature vectors
    :param normalization_crop_pad: Specify in xyz how much to crop input before the normalizing
    :return: RGB map
    """
    assert (data.ndim == 5) and (data.shape[0] == 1)
    crop_pad = normalization_crop_pad
    if crop_pad is not None:
        assert len(crop_pad) == 3
    data_np = convert.to_np(data).astype(np.float32)
    rgbmap = data_np[:, 0:3, ...]

    rgbmap_cropped = rgbmap
    if crop_pad is not None:
        rgbmap_cropped = data_np[
            :,
            :,
            crop_pad[0] : -crop_pad[0],
            crop_pad[1] : -crop_pad[1],
            crop_pad[2] : -crop_pad[2],
        ]

    if per_channel_normalization:
        for channel in range(3):
            rgbmap[:, channel, ...] -= np.min(rgbmap_cropped[:, channel, ...])
            rgbmap[:, channel, ...] /= np.max(rgbmap_cropped[:, channel, ...])
    else:
        rgbmap -= np.min(rgbmap_cropped)
        rgbmap /= np.max(rgbmap_cropped)

    result = convert.astype(rgbmap, data)
    return result


@builder.register("vec_to_affs")
@typechecked
def vec_to_affs(
    vec: torch.Tensor,
    edges: Sequence[Sequence[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # assume CXYZ
    delta_d: float = 1.5,
) -> torch.Tensor:

    assert vec.ndimension() >= 4
    assert len(edges) > 0

    affs = []
    for edge in edges:
        pair = tensor_ops.get_disp_pair(vec, edge)
        aff = _compute_affinity(pair[0], pair[1], delta_d=delta_d)  #
        pad = []
        for e in reversed(edge):
            if e > 0:
                pad.extend([e, 0])
            else:
                pad.extend([0, abs(e)])
        affs.append(F.pad(aff, pad))

    assert len(affs) > 0
    for aff in affs:
        assert affs[0].size() == aff.size()

    return torch.cat(affs, dim=-4)


def _compute_affinity(
    embd1: torch.Tensor,
    embd2: torch.Tensor,
    dim: int = -4,
    keepdims: bool = True,
    delta_d: float = 1.5,
) -> torch.Tensor:
    """Compute an affinity map from a pair of embeddings."""
    norm = torch.norm(embd1 - embd2, p=1, dim=dim, keepdim=keepdims)
    margin = (2 * delta_d - norm) / (2 * delta_d)
    zero = torch.zeros(1).to(embd1.device, dtype=embd1.dtype)
    result = torch.max(zero, margin) ** 2
    return result


@builder.register("vec_to_affs_v1")
@typechecked
def vec_to_affs_v1(
    embeddings: torch.Tensor,
    offsets: Sequence[int] = (1, 1, 1),
    delta_mult: int = 15000,
) -> torch.Tensor:
    # Tri's naive implementation
    metric_out = np.zeros(shape=(3,) + embeddings.shape[1:])
    # compute mean-square
    metric_out[0, offsets[0] :, :, :] = (
        (embeddings[:, offsets[0] :, :, :] - embeddings[:, : -offsets[0], :, :]) ** 2
    ).mean(axis=0)
    metric_out[1, :, offsets[1] :, :] = (
        (embeddings[:, :, offsets[1] :, :] - embeddings[:, :, : -offsets[1], :]) ** 2
    ).mean(axis=0)
    metric_out[2, :, :, offsets[2] :] = (
        (embeddings[:, :, :, offsets[2] :] - embeddings[:, :, :, : -offsets[2]]) ** 2
    ).mean(axis=0)
    metric_out *= delta_mult
    # convert to affinities and torch.tensor
    metric_out[metric_out > 1] = 1
    metric_out_ = torch.Tensor(1.0 - metric_out)
    return metric_out_
