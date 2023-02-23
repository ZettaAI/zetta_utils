# pylint: disable=too-many-locals
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import attrs
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


def get_aced_match_offsets_naive(
    non_tissue: torch.Tensor,
    misalignment_mask_zm1: torch.Tensor,
    misalignment_mask_zm2: Optional[torch.Tensor] = None,
    misalignment_mask_zm3: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    match_offsets = torch.ones_like(non_tissue, dtype=torch.int) * -1
    match_offsets[non_tissue] = 0

    misalignment_mask_map = {
        1: misalignment_mask_zm1,
        2: misalignment_mask_zm2,
        3: misalignment_mask_zm3,
    }

    for offset in sorted(misalignment_mask_map.keys()):
        unmatched_locations = match_offsets == -1
        if unmatched_locations.sum() == 0:
            break
        if misalignment_mask_map[offset] is not None:
            current_match_locations = misalignment_mask_map[offset] == 0
            match_offsets[unmatched_locations * current_match_locations] = offset

    match_offsets[match_offsets == -1] = 0
    result = match_offsets.byte()
    return result


def get_aced_match_offsets(
    tissue_mask: torch.Tensor,
    misalignment_masks: dict[str, torch.Tensor],
    pairwise_fields: dict[str, torch.Tensor],
    pairwise_fields_inv: dict[str, torch.Tensor],
    max_dist: int,
) -> dict[str, torch.Tensor]:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tissue_mask_zcxy = einops.rearrange(tissue_mask, "1 X Y Z -> Z 1 X Y").to(device)
    misalignment_masks_zcxy = {
        k: einops.rearrange(v, "1 X Y Z -> Z 1 X Y").to(device)
        for k, v in misalignment_masks.items()
    }
    pairwise_fields_zcxy = {
        k: einops.rearrange(v, "C X Y Z -> Z C X Y")
        .field()  # type: ignore
        .from_pixels()
        .to(device)
        for k, v in pairwise_fields.items()
    }
    pairwise_fields_inv_zcxy = {
        k: einops.rearrange(v, "C X Y Z -> Z C X Y")
        .field()  # type: ignore
        .from_pixels()
        .to(device)
        for k, v in pairwise_fields_inv.items()
    }

    fwd_outcome = _perform_match_fwd_pass(
        tissue_mask_zcxy=tissue_mask_zcxy,
        misalignment_masks_zcxy=misalignment_masks_zcxy,
        pairwise_fields_zcxy=pairwise_fields_zcxy,
        pairwise_fields_inv_zcxy=pairwise_fields_inv_zcxy,
        max_dist=max_dist,
    )
    sector_length_after_zcxy = _perform_match_bwd_pass(
        match_offsets_inv_zcxy=fwd_outcome.match_offsets_inv_zcxy,
        pairwise_fields_inv_zcxy=pairwise_fields_inv_zcxy,
        max_dist=max_dist,
    )
    img_mask_zcxy, aff_mask_zcxy = _get_masks(
        sector_length_up_to_zcxy=fwd_outcome.sector_length_up_to_zcxy,
        sector_length_after_zcxy=sector_length_after_zcxy,
        max_dist=max_dist,
    )
    result = {
        "match_offsets": fwd_outcome.match_offsets_zcxy,
        "img_mask": img_mask_zcxy,
        "aff_mask": aff_mask_zcxy,
    }
    result = {k: einops.rearrange(v, "Z C X Y -> C X Y Z") for k, v in result.items()}
    return result


def _get_masks(
    sector_length_up_to_zcxy: torch.Tensor,
    sector_length_after_zcxy: torch.Tensor,
    max_dist: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    img_mask_zcxy = (sector_length_up_to_zcxy + sector_length_after_zcxy) < max_dist
    aff_mask_zcxy = (sector_length_up_to_zcxy == 0) * (img_mask_zcxy == 0)
    return img_mask_zcxy, aff_mask_zcxy


def _perform_match_bwd_pass(
    match_offsets_inv_zcxy: torch.Tensor,
    pairwise_fields_inv_zcxy: dict[str, torch.Tensor],
    max_dist: int,
):
    sector_length_after_zcxy = torch.zeros_like(match_offsets_inv_zcxy)
    num_sections = match_offsets_inv_zcxy.shape[0]
    for i in range(num_sections - 1, -1, -1):
        for offset in range(1, max_dist + 1):
            j = i + offset
            if j >= num_sections:
                break

            this_pairwise_field_inv = pairwise_fields_inv_zcxy[str(-offset)][j : j + 1]

            this_offset_sector_length = this_pairwise_field_inv.sample(  # type: ignore
                sector_length_after_zcxy[j].float(), mode="nearest"
            ).int()
            this_offset_sector_length[match_offsets_inv_zcxy[i] != offset] = 0
            this_offset_sector_length[match_offsets_inv_zcxy[i] == offset] += 1

            sector_length_after_zcxy[i] = torch.max(
                sector_length_after_zcxy[i], this_offset_sector_length
            )
    return sector_length_after_zcxy


@attrs.mutable
class _FwdPassOutcome:
    sector_length_up_to_zcxy: torch.Tensor
    match_offsets_zcxy: torch.Tensor
    match_offsets_inv_zcxy: torch.Tensor


def _perform_match_fwd_pass(
    tissue_mask_zcxy: torch.Tensor,
    misalignment_masks_zcxy: dict[str, torch.Tensor],
    pairwise_fields_zcxy: dict[str, torch.Tensor],
    pairwise_fields_inv_zcxy: dict[str, torch.Tensor],
    max_dist: int,
) -> _FwdPassOutcome:
    num_sections = tissue_mask_zcxy.shape[0]

    sector_length_up_to_zcxy = torch.zeros_like(tissue_mask_zcxy).int()
    match_offsets_zcxy = torch.zeros_like(tissue_mask_zcxy).int()
    match_offsets_inv_zcxy = torch.zeros_like(tissue_mask_zcxy).int()

    for i in range(1, num_sections):
        offset_scores = torch.zeros(
            (max_dist, 1, tissue_mask_zcxy.shape[-2], tissue_mask_zcxy.shape[-1]),
            dtype=torch.float32,
            device=tissue_mask_zcxy.device,
        )

        offset_sector_lengths = torch.zeros(
            (max_dist, 1, tissue_mask_zcxy.shape[-2], tissue_mask_zcxy.shape[-1]),
            dtype=torch.int32,
            device=tissue_mask_zcxy.device,
        )

        for offset in range(1, max_dist + 1):
            j = i - offset
            if j < 0:
                break

            this_misalignment_mask = misalignment_masks_zcxy[str(-offset)][i]
            this_tissue_mask = tissue_mask_zcxy[j]
            this_pairwise_field = pairwise_fields_zcxy[str(-offset)][i : i + 1]

            offset_sector_lengths[offset - 1] = (
                this_pairwise_field.sample(  # type: ignore
                    sector_length_up_to_zcxy[j].float(), mode="nearest"
                ).int()
                + 1
            )

            offset_sector_lengths[offset - 1][this_tissue_mask == 0] = 0
            offset_sector_lengths[offset - 1][this_misalignment_mask] = 0
            offset_sector_length_scores = offset_sector_lengths[offset - 1] / (
                offset_sector_lengths[offset - 1].max(0)[0] + 1e-4
            )
            assert offset_sector_length_scores.max() <= 1.0

            offset_scores[offset - 1] = tissue_mask_zcxy[j] * 100
            offset_scores[offset - 1] += (misalignment_masks_zcxy[str(-offset)][i] == 0) * 10
            offset_scores[offset - 1] += offset_sector_length_scores
            offset_scores[offset - 1] += (max_dist - offset) / 100

        chosen_offset_scores, chosen_offsets = offset_scores.max(0)
        passable_choices = chosen_offset_scores >= 100
        match_offsets_zcxy[i][passable_choices] = chosen_offsets[passable_choices].int() + 1

        # sector_length_up_to_zcxy[i] = offset_sector_lengths[chosen_offsets]
        # TODO: how do vectorize this?
        for choice in range(0, max_dist):
            this_match_locations = chosen_offsets == choice
            sector_length_up_to_zcxy[i][this_match_locations] = offset_sector_lengths[choice][
                this_match_locations
            ]

        for offset in range(1, max_dist + 1):
            j = i - offset
            this_offset_matches = match_offsets_zcxy[i] == offset
            # Discard non-aligned matches for bwd pass
            this_offset_matches[chosen_offset_scores < 110] = 0
            if this_offset_matches.sum() > 0:
                this_inv_field = pairwise_fields_inv_zcxy[str(-offset)][i : i + 1]
                this_offset_matches_inv = this_inv_field.sample(  # type: ignore
                    this_offset_matches.float(), mode="nearest"
                ).int()
                this_offset_matches_inv[tissue_mask_zcxy[j] == 0] = 0
                match_offsets_inv_zcxy[i][this_offset_matches_inv != 0] = offset

    return _FwdPassOutcome(
        sector_length_up_to_zcxy=sector_length_up_to_zcxy,
        match_offsets_zcxy=match_offsets_zcxy,
        match_offsets_inv_zcxy=match_offsets_inv_zcxy,
    )
