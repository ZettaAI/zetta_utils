# pylint: disable = no-self-use
from __future__ import annotations

from typing import Sequence

import attrs
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.tensor_ops import convert


@typechecked
def create_mapping(trgt: npt.NDArray, splt: npt.NDArray, mask: npt.NDArray) -> list[list[int]]:
    trgt, splt = trgt.astype(np.uint64), splt.astype(np.uint64)
    encoded = (2 ** 32) * trgt + splt
    encoded[mask == 0] = 0
    unq = np.unique(encoded)
    mapping: dict[int, list[int]] = {}
    for unq_id in unq:
        trgt_id, splt_id = int(unq_id // (2 ** 32)), int(unq_id % (2 ** 32))
        mapping[trgt_id] = mapping.get(trgt_id, []) + [splt_id]
    result = list(mapping.values())
    return result


@builder.register("MeanLoss")
@typechecked
@attrs.mutable
class MeanLoss(nn.Module):
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.001
    delta_v: float = 0.0
    delta_d: float = 1.5
    recompute_ext: bool = False

    def __attrs_pre_init__(self):
        super().__init__()

    def forward(
        self,
        embd: torch.Tensor,
        trgt: torch.Tensor,
        mask: torch.Tensor,
        splt: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        :param embd: Embeddings
        :param trgt: Target segmentation
        :param mask: Segmentation mask
        :param splt: Connected components of the target segmentation
        """
        groups = None
        if self.recompute_ext:
            assert splt is not None
            trgt_np = np.squeeze(convert.to_np(trgt))
            splt_np = np.squeeze(convert.to_np(splt))
            mask_np = np.squeeze(convert.to_np(mask))
            groups = create_mapping(trgt_np, splt_np, mask_np)
            trgt = splt

        trgt = trgt.to(torch.int)
        trgt *= (mask > 0).to(torch.int)

        # Unique nonzero IDs
        ids = np.unique(convert.to_np(trgt))
        ids = ids[ids != 0].tolist()

        mext = self.compute_ext_matrix(ids, groups, self.recompute_ext)
        vecs = self.generate_vecs(embd, trgt, ids)
        means = [torch.mean(vec, dim=0) for vec in vecs]
        weights = [1.0] * len(means)

        # Compute loss
        loss_int = self.compute_loss_int(vecs, means, weights, embd.device)
        loss_ext = self.compute_loss_ext(means, weights, mext, embd.device)
        loss_nrm = self.compute_loss_nrm(means, embd.device)

        result = (self.alpha * loss_int) + (self.beta * loss_ext) + (self.gamma * loss_nrm)
        return result

    def compute_loss_int(
        self,
        vecs: list[torch.Tensor],
        means: list[torch.Tensor],
        weights: list[float],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the internal term of the loss."""
        assert len(vecs) == len(means) == len(weights)
        zero = lambda: torch.zeros(1).to(device).squeeze()
        loss = zero()
        for vec, mean, weight in zip(vecs, means, weights):
            margin = torch.norm(vec - mean, p=1, dim=1) - self.delta_v
            loss += weight * torch.mean(torch.max(margin, zero()) ** 2)
        result = loss / max(1.0, len(vecs))
        return result

    def compute_loss_ext(
        self,
        means: list[torch.Tensor],
        weights: list[float],
        mext: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute the external term of the loss."""
        assert len(means) == len(weights)
        zero = lambda: torch.zeros(1).to(device).squeeze()
        loss = zero()
        count = len(means)
        if (count > 1) and (mext is not None):
            means1 = torch.stack(means).unsqueeze(0)  # 1 x N x Dim
            means2 = torch.stack(means).unsqueeze(1)  # N x 1 x Dim
            margin = 2 * self.delta_d - torch.norm(means2 - means1, p=1, dim=2)
            margin = margin[mext]
            loss = torch.sum(torch.max(margin, zero()) ** 2)
        result = loss / max(1.0, count * (count - 1))
        return result

    def compute_loss_nrm(self, means: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Compute the regularization term of the loss."""
        zero = lambda: torch.zeros(1).to(device).squeeze()
        loss = zero()
        if len(means) > 0:
            loss += torch.mean(torch.norm(torch.stack(means), p=1, dim=1))
        result = loss
        return result

    def generate_vecs(
        self,
        embd: torch.Tensor,
        trgt: torch.Tensor,
        ids: Sequence[int],
    ) -> list[torch.Tensor]:
        """
        Generate a list of vectorized embeddings for each ground truth object.
        """
        result = []
        for obj_id in ids:
            obj = torch.nonzero(trgt == obj_id)
            z, y, x = obj[:, -3], obj[:, -2], obj[:, -1]
            vec = embd[0, :, z, y, x].transpose(0, 1)  # Count x Dim
            result.append(vec)
        return result

    def compute_ext_matrix(
        self,
        ids: Sequence[int],
        groups: Sequence[Sequence[int]] | None = None,
        recompute_ext: bool = False,
    ) -> torch.Tensor | None:
        """
        Compute a matrix that indicates the presence of 'external' interaction
        between objects.
        """
        num_ids = len(ids)
        mext = torch.ones((num_ids, num_ids)) - torch.eye(num_ids)

        # Recompute external matrix
        if recompute_ext:
            assert groups is not None
            idmap = {x: i for i, x in enumerate(ids)}
            for group in groups:
                for i, id_i in enumerate(group):
                    for id_j in group[i + 1 :]:
                        mext[idmap[id_i], idmap[id_j]] = 0
                        mext[idmap[id_j], idmap[id_i]] = 0

        # Safeguard
        if mext.sum().item() == 0:
            return None

        result = mext.to(torch.bool)
        return result
