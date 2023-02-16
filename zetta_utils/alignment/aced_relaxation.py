# pylint: disable=too-many-locals
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import einops
import metroem
import torch
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, log

logger = log.get_logger("zetta_utils")


def compute_aced_loss(
    pfields: Dict[Tuple[int, int], torch.Tensor],
    afields: List[torch.Tensor],
    match_offsets: List[torch.Tensor],
    rigidity_weight: float,
    rigidity_masks: torch.Tensor,
) -> torch.Tensor:
    intra_loss = 0
    inter_loss = 0

    for i in range(1, len(afields)):
        offset = 1
        while (i, i - offset) in pfields:
            inter_loss_map = (
                pfields[(i, i - offset)]
                .from_pixels()(  # type: ignore
                    afields[i - offset].from_pixels()  # type: ignore
                )
                .pixels()
                - afields[i]
            )
            inter_loss_map_mask = (
                afields[i]
                .from_pixels()((match_offsets[i] == offset).float())  # type: ignore
                .squeeze()
                > 0.0
            )
            this_inter_loss = (inter_loss_map[..., inter_loss_map_mask] ** 2).sum()
            inter_loss += this_inter_loss
            offset += 1

        intra_loss_map = metroem.loss.rigidity(afields[i])
        this_intra_loss = intra_loss_map[rigidity_masks[i].squeeze()].sum()
        intra_loss += this_intra_loss

    loss = inter_loss + rigidity_weight * intra_loss
    print(inter_loss, rigidity_weight * intra_loss, loss)
    return loss  # type: ignore


def _get_opt_range(fix: Literal["first", "last", "both"] | None, num_sections: int):
    if fix is None:
        opt_range = range(num_sections)
    elif fix == "first":
        opt_range = range(1, num_sections)
    elif fix == "last":
        opt_range = range(num_sections - 1)
    else:
        assert fix == "both"
        opt_range = range(1, num_sections - 1)
    return opt_range


@builder.register("perform_aced_relaxation")
def perform_aced_relaxation(
    match_offsets: torch.Tensor,
    field_zm1: torch.Tensor,
    field_zm2: Optional[torch.Tensor] = None,
    field_zm3: Optional[torch.Tensor] = None,
    rigidity_masks: torch.Tensor | None = None,
    num_iter=100,
    lr=0.3,
    rigidity_weight=10.0,
    fix: Optional[Literal["first", "last", "both"]] = "first",
) -> torch.Tensor:
    max_displacement = 0.0
    for field in [field_zm1, field_zm2, field_zm2]:
        if field is not None:
            max_displacement = max(max_displacement, field.abs().max().item())
    if (match_offsets != 0).sum() == 0 or max_displacement < 0.01:
        return torch.zeros_like(field_zm1)
    match_offsets_zcxy = einops.rearrange(match_offsets, "C X Y Z -> Z C X Y").cuda()

    if rigidity_masks is not None:
        rigidity_masks_zcxy = einops.rearrange(rigidity_masks, "C X Y Z -> Z C X Y").cuda()
    else:
        rigidity_masks_zcxy = torch.ones_like(match_offsets_zcxy)

    num_sections = match_offsets_zcxy.shape[0]
    assert num_sections > 1, "Can't relax blocks with just one section"
    field_map = {
        1: field_zm1,
        2: field_zm2,
        3: field_zm3,
    }
    pfields: Dict[Tuple[int, int], torch.Tensor] = {}
    for offset, field in field_map.items():
        if field is not None:
            field_zcxy = einops.rearrange(field, "C X Y Z -> Z C X Y").field()  # type: ignore

            for i in range(num_sections):
                pfields[(i, i - offset)] = field_zcxy[i : i + 1].cuda()

    afields = [
        torch.zeros((1, 2, match_offsets_zcxy.shape[2], match_offsets_zcxy.shape[3]))
        .cuda()
        .field()  # type: ignore
        for _ in range(num_sections)
    ]

    opt_range = _get_opt_range(fix=fix, num_sections=num_sections)
    for i in opt_range:
        afields[i].requires_grad = True

    optimizer = torch.optim.Adam(
        [afields[i] for i in opt_range],
        lr=lr,
    )

    for i in range(num_iter):
        loss = compute_aced_loss(
            pfields=pfields,
            afields=afields,
            rigidity_masks=rigidity_masks_zcxy,
            match_offsets=[match_offsets_zcxy[i] for i in range(num_sections)],
            rigidity_weight=rigidity_weight,
        )
        if loss < 0.005:
            break
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result_xy = torch.cat(afields, 0)
    result = einops.rearrange(result_xy, "Z C X Y -> C X Y Z")
    return result


def get_aced_match_offsets(
    non_tissue: torch.Tensor,
    misd_mask_zm1: torch.Tensor,
    misd_mask_zm2: Optional[torch.Tensor] = None,
    misd_mask_zm3: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    match_offsets = torch.ones_like(non_tissue, dtype=torch.int) * -1
    match_offsets[non_tissue] = 0

    misd_mask_map = {
        1: misd_mask_zm1,
        2: misd_mask_zm2,
        3: misd_mask_zm3,
    }

    for offset in sorted(misd_mask_map.keys()):
        unmatched_locations = match_offsets == -1
        if unmatched_locations.sum() == 0:
            break
        if misd_mask_map[offset] is not None:
            current_match_locations = misd_mask_map[offset] == 0
            match_offsets[unmatched_locations * current_match_locations] = offset

    match_offsets[match_offsets == -1] = 0
    result = match_offsets.byte()
    return result
