from typing import Sequence

import attrs
import einops
import numpy as np
import torch
from torch.nn import functional as F
from typeguard import typechecked

from zetta_utils import builder, convnet


@builder.register("AffinitiesInferencer")
@typechecked
@attrs.frozen
class AffinitiesInferencer:
    # Input uint8  [   0 .. 255]
    # Output float [   0 .. 255]

    # Don't create the model during initialization for efficient serialization
    model_path: str
    output_channels: Sequence[int]

    bg_mask_channel: int | None = None
    bg_mask_threshold: float = 0.0
    bg_mask_invert_threshold: bool = False

    def __call__(
        self,
        image: torch.Tensor,
        image_mask: torch.Tensor,
        output_mask: torch.Tensor,
    ) -> torch.Tensor:

        if image.dtype == torch.uint8:
            data_in = image.float() / 255.0  # [0.0 .. 1.0]
        else:
            raise ValueError(f"Unsupported image dtype: {image.dtype}")

        # mask input
        data_in = data_in * image_mask
        data_in = einops.rearrange(data_in, "C X Y Z -> C Z Y X")
        data_in = data_in.unsqueeze(0).float()

        data_out = convnet.utils.load_and_run_model(path=self.model_path, data_in=data_in)

        # Extract requested channels
        arrays = []
        for channel in self.output_channels:
            arrays.append(data_out[:, channel, ...])
        if self.bg_mask_channel is not None:
            arrays.append(data_out[:, self.bg_mask_channel, ...])
        data_out = torch.Tensor(np.stack(arrays, axis=1)[0])

        # mask output with bg_mask
        num_channels = len(self.output_channels)
        output = data_out[0:num_channels, :, :, :]
        if self.bg_mask_channel is not None:
            if self.bg_mask_invert_threshold:
                bg_mask = data_out[num_channels:, :, :, :] > self.bg_mask_threshold
            else:
                bg_mask = data_out[num_channels:, :, :, :] < self.bg_mask_threshold
            output = torch.Tensor(output) * bg_mask

        # mask output
        output = einops.rearrange(output, "C Z Y X -> C X Y Z")
        output = output * output_mask

        return output


@builder.register("vec2aff")
@typechecked
def vec2aff(
    vec: torch.Tensor,
    edges: Sequence[Sequence[int]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),  # assume CXYZ
    delta_d: float = 1.5,
) -> torch.Tensor:

    assert vec.ndimension() >= 4
    assert len(edges) > 0

    affs = []
    for edge in edges:
        aff = compute_affinity(*(get_pair(vec, edge)), delta_d=delta_d)  # type: ignore
        pad = ()
        for e in reversed(edge):
            if e > 0:
                pad += (e, 0) # type: ignore
            else:
                pad += (0, abs(e)) # type: ignore
        affs.append(F.pad(aff, pad))

    assert len(affs) > 0
    for aff in affs:
        assert affs[0].size() == aff.size()

    return torch.cat(affs, dim=-4)


def compute_affinity(
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


def get_pair(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    arr1 = arr[
        ..., os1[0] : shape[0] - os2[0], os1[1] : shape[1] - os2[1], os1[2] : shape[2] - os2[2]
    ]
    arr2 = arr[
        ..., os2[0] : shape[0] - os1[0], os2[1] : shape[1] - os1[1], os2[2] : shape[2] - os1[2]
    ]
    return arr1, arr2


@builder.register("vec2aff_v1")
@typechecked
def vec2aff_v1(
    embeddings,
    offsets=(1, 1, 1),
    delta_mult=15000,
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
    # convert to affinities
    metric_out[metric_out > 1] = 1
    metric_out_ = torch.Tensor(1.0 - metric_out)
    return metric_out_
