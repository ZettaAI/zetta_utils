# pylint: disable=too-many-locals
from __future__ import annotations

from typing import Dict, List, Literal, Optional

import attrs
import einops

# import metroem
import torch
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, log

from .field import get_rigidity_map_zcxy, invert_field

logger = log.get_logger("zetta_utils")


def compute_aced_loss_new(
    pfields_raw: Dict[int, torch.Tensor],
    afields: List[torch.Tensor],
    match_offsets: List[torch.Tensor],
    rigidity_weight: float,
    rigidity_masks: torch.Tensor,
    max_dist: int,
    min_rigidity_multiplier: float,
) -> torch.Tensor:
    intra_loss = 0
    inter_loss = 0
    afields_cat = torch.cat(afields)
    match_offsets_cat = torch.stack(match_offsets)

    match_offsets_warped = {
        offset: afields_cat((match_offsets_cat == offset).float()) > 0.7  # type: ignore
        for offset in range(1, max_dist + 1)
    }
    inter_loss = 0
    for offset in range(1, max_dist + 1):
        inter_expectation = afields_cat[:-offset](pfields_raw[offset][offset:])  # type: ignore
        inter_loss_map = inter_expectation - afields_cat[offset:]

        inter_loss_map_mask = match_offsets_warped[offset].squeeze()[offset:]
        this_inter_loss = (inter_loss_map ** 2).sum(1)[..., inter_loss_map_mask].sum()
        inter_loss += this_inter_loss

    intra_loss_map = get_rigidity_map_zcxy(afields_cat.pixels())  # type: ignore
    with torch.no_grad():
        rigidity_masks_warped = afields_cat(rigidity_masks.float())  # type: ignore
        rigidity_masks_warped[
            rigidity_masks_warped < min_rigidity_multiplier
        ] = min_rigidity_multiplier
    intra_loss = (intra_loss_map * rigidity_masks_warped.squeeze()).sum()
    loss = inter_loss + rigidity_weight * intra_loss / (
        afields[0].shape[-1] * afields[0].shape[-1] / 4
    )

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
def perform_aced_relaxation(  # pylint: disable=too-many-branches
    match_offsets: torch.Tensor,
    pfields: dict[str, torch.Tensor],
    rigidity_masks: torch.Tensor | None = None,
    first_section_fix_field: torch.Tensor | None = None,
    last_section_fix_field: torch.Tensor | None = None,
    min_rigidity_multiplier: float = 0.0,
    num_iter=100,
    lr=0.3,
    rigidity_weight=10.0,
    fix: Optional[Literal["first", "last", "both"]] = "first",
    max_dist: int = 2,
    grad_clip: float | None = None,
) -> torch.Tensor:
    assert "-1" in pfields

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    max_displacement = max([field.abs().max().item() for field in pfields.values()])

    if (match_offsets != 0).sum() == 0 or max_displacement < 0.01:
        return torch.zeros_like(pfields["-1"])

    match_offsets_zcxy = einops.rearrange(match_offsets, "C X Y Z -> Z C X Y").to(device)

    if rigidity_masks is not None:
        rigidity_masks_zcxy = einops.rearrange(rigidity_masks, "C X Y Z -> Z C X Y").to(device)
    else:
        rigidity_masks_zcxy = torch.ones_like(match_offsets_zcxy)

    num_sections = match_offsets_zcxy.shape[0]
    assert num_sections > 1, "Can't relax blocks with just one section"

    pfields_raw: Dict[int, torch.Tensor] = {}

    for offset_str, field in pfields.items():
        offset = -int(offset_str)
        pfields_raw[offset] = (
            einops.rearrange(field, "C X Y Z -> Z C X Y")
            .field()  # type: ignore
            .to(device)
            .from_pixels()
        )

    if first_section_fix_field is not None:
        assert fix in ["first", "both"]

        first_section_fix_field_zcxy = (
            einops.rearrange(first_section_fix_field, "C X Y Z -> Z C X Y")
            .field()  # type: ignore
            .to(device)
            .from_pixels()
        )
        for offset in range(1, max_dist + 1):
            pfields_raw[offset][offset] = first_section_fix_field_zcxy(pfields_raw[offset][offset])

    if last_section_fix_field is not None:
        assert fix in ["last", "both"]

        last_section_fix_field_inv = invert_field(last_section_fix_field.to(device))
        last_section_fix_field_inv_zcxy = (
            einops.rearrange(last_section_fix_field_inv, "C X Y Z -> Z C X Y")
            .field()  # type: ignore
            .to(device)
            .from_pixels()
        )

        for offset in range(1, max_dist + 1):
            pfields_raw[offset][-1] = pfields_raw[offset][-1](
                last_section_fix_field_inv_zcxy  # type: ignore
            )

    afields = [
        torch.zeros((1, 2, match_offsets_zcxy.shape[2], match_offsets_zcxy.shape[3]))
        .to(device)
        .field()  # type: ignore
        .from_pixels()
        for _ in range(num_sections)
    ]

    opt_range = _get_opt_range(fix=fix, num_sections=num_sections)
    for i in opt_range:
        afields[i].requires_grad = True

    optimizer = torch.optim.Adam(
        [afields[i] for i in opt_range],
        lr=lr,
    )

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
        for i in range(num_iter):
            loss_new = compute_aced_loss_new(
                pfields_raw=pfields_raw,
                afields=afields,
                rigidity_masks=rigidity_masks_zcxy,
                match_offsets=[match_offsets_zcxy[i] for i in range(num_sections)],
                rigidity_weight=rigidity_weight,
                max_dist=max_dist,
                min_rigidity_multiplier=min_rigidity_multiplier,
            )
            loss = loss_new
            if i % 100 == 0:
                logger.info(f"Iter {i} loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(afields, max_norm=grad_clip)
            optimizer.step()

    result_xy = torch.cat(afields, 0).pixels()  # type: ignore
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

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
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
            # pairwise_fields_zcxy=pairwise_fields_zcxy,
            pairwise_fields_inv_zcxy=pairwise_fields_inv_zcxy,
            max_dist=max_dist,
        )
        sector_length_after_zcxy = _perform_match_bwd_pass(
            match_offsets_inv_zcxy=fwd_outcome.match_offsets_inv_zcxy,
            pairwise_fields_zcxy=pairwise_fields_zcxy,
            max_dist=max_dist,
        )
        img_mask_zcxy, aff_mask_zcxy = _get_masks(
            sector_length_before_zcxy=fwd_outcome.sector_length_before_zcxy,
            sector_length_after_zcxy=sector_length_after_zcxy,
            # match_offsets_zcxy=fwd_outcome.match_offsets_zcxy,
            # pairwise_fields_inv_zcxy=pairwise_fields_inv_zcxy,
            max_dist=max_dist,
            tissue_mask_zcxy=tissue_mask_zcxy,
        )
    result = {
        "match_offsets": fwd_outcome.match_offsets_zcxy,
        "img_mask": img_mask_zcxy,
        "aff_mask": aff_mask_zcxy,
        "sector_length_after": sector_length_after_zcxy,
        "sector_length_before": fwd_outcome.sector_length_before_zcxy,
    }
    result = {k: einops.rearrange(v, "Z C X Y -> C X Y Z") for k, v in result.items()}
    return result


@attrs.mutable
class _FwdPassOutcome:
    sector_length_before_zcxy: torch.Tensor
    match_offsets_zcxy: torch.Tensor
    match_offsets_inv_zcxy: torch.Tensor


def _perform_match_fwd_pass(
    tissue_mask_zcxy: torch.Tensor,
    misalignment_masks_zcxy: dict[str, torch.Tensor],
    pairwise_fields_inv_zcxy: dict[str, torch.Tensor],
    max_dist: int,
) -> _FwdPassOutcome:
    num_sections = tissue_mask_zcxy.shape[0]

    sector_length_before_zcxy = torch.zeros_like(tissue_mask_zcxy).int()
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

            this_pairwise_field_inv = pairwise_fields_inv_zcxy[str(-offset)][i : i + 1]

            tgt_tissue_mask = this_pairwise_field_inv.sample(  # type: ignore
                tissue_mask_zcxy[j].float(),
                mode="nearest",
            ).int()

            this_tissue_mask = tissue_mask_zcxy[i] * tgt_tissue_mask

            this_misalignment_mask = misalignment_masks_zcxy[str(-offset)][i]

            offset_sector_lengths[offset - 1] = (
                this_pairwise_field_inv.sample(  # type: ignore
                    sector_length_before_zcxy[j].float(),
                    mode="nearest",
                ).int()
                + 1
            )

            # Offset lengths score prioritizes longer match chain
            # misalignmened and non-tissue correspondences are not prioritized by it
            offset_sector_lengths[offset - 1][this_tissue_mask == 0] = 0
            offset_sector_lengths[offset - 1][this_misalignment_mask != 0] = 0
            offset_sector_length_scores = offset_sector_lengths[offset - 1] / 1e5
            assert offset_sector_length_scores.max() <= 1.0

            offset_scores[offset - 1] = this_tissue_mask * 100
            offset_scores[offset - 1] += (misalignment_masks_zcxy[str(-offset)][i] == 0) * 10
            offset_scores[offset - 1] += offset_sector_length_scores
            offset_scores[offset - 1] += (max_dist - offset) / 1e7

        chosen_offset_scores, chosen_offsets = offset_scores.max(0)
        passable_choices = chosen_offset_scores >= 110
        match_offsets_zcxy[i][passable_choices] = chosen_offsets[passable_choices].int() + 1
        # match_offsets_zcxy[i] = this_tissue_mask

        # sector_length_before_zcxy[i] = offset_sector_lengths[chosen_offsets]
        # TODO: how do vectorize this?
        for choice in range(0, max_dist):
            this_match_locations = chosen_offsets == choice
            sector_length_before_zcxy[i][this_match_locations] = offset_sector_lengths[choice][
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
                match_offsets_inv_zcxy[j][this_offset_matches_inv != 0] = offset

    return _FwdPassOutcome(
        sector_length_before_zcxy=sector_length_before_zcxy,
        match_offsets_zcxy=match_offsets_zcxy,
        match_offsets_inv_zcxy=match_offsets_inv_zcxy,
    )


def _get_masks(
    sector_length_before_zcxy: torch.Tensor,
    sector_length_after_zcxy: torch.Tensor,
    # pairwise_fields_inv_zcxy: dict[str, torch.Tensor],
    # match_offsets_zcxy: torch.Tensor,
    tissue_mask_zcxy: torch.Tensor,
    max_dist: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # num_sections = sector_length_before_zcxy.shape[0]

    # img_mask_zcxy = (sector_length_before_zcxy + sector_length_after_zcxy) < max_dist
    # aff_mask_zcxy = (sector_length_before_zcxy == 0) * (img_mask_zcxy == 0)

    img_mask_zcxy = (sector_length_before_zcxy + sector_length_after_zcxy) < max_dist

    aff_mask_zcxy = (sector_length_before_zcxy == 0) * (sector_length_after_zcxy >= max_dist)
    # TODO: fix
    # aff_mask_zcxy[1:] += (sector_length_after_zcxy[:-1] == 0) * (
    #    sector_length_before_zcxy[:-1] >= max_dist
    # )
    # TODO: Decide whether we want this
    # for i in range(1, num_sections):
    #    for offset in range(1, max_dist + 1):

    #        j = i - offset
    #        this_offset_matches = match_offsets_zcxy[i] == offset

    #        if this_offset_matches.sum() > 0:
    #            this_inv_field = pairwise_fields_inv_zcxy[str(-offset)][i : i + 1]
    #            this_sector_length_after_from_j = this_inv_field.sample(  # type: ignore
    #                sector_length_after_zcxy[j].float(), mode="nearest"
    #            ).int()
    #            this_sector_length_before_from_j = this_inv_field.sample(  # type: ignore
    #                sector_length_before_zcxy[j].float(), mode="nearest"
    #            ).int()

    #            back_connected_locations = sector_length_before_zcxy[i] == (
    #                this_sector_length_before_from_j + 1
    #            )
    #            mid_connected_locations = sector_length_after_zcxy[i] == (
    #                this_sector_length_after_from_j - 1
    #            )
    #            dangling_tail_locations = (
    #                back_connected_locations * (mid_connected_locations == 0) *
    #                this_offset_matches
    #            )

    #            img_mask_zcxy[i][dangling_tail_locations] = True
    #            aff_mask_zcxy[i][dangling_tail_locations] = False
    #            if i + i < num_sections:
    #               aff_mask_zcxy[i + 1][dangling_tail_locations] = False

    img_mask_zcxy[0] = False
    aff_mask_zcxy[0] = False
    aff_mask_zcxy[-1][img_mask_zcxy[-1] != 0] = 1
    img_mask_zcxy[-1] = 0
    img_mask_zcxy[tissue_mask_zcxy == 0] = 1
    return img_mask_zcxy, aff_mask_zcxy


def _perform_match_bwd_pass(
    match_offsets_inv_zcxy: torch.Tensor,
    pairwise_fields_zcxy: dict[str, torch.Tensor],
    max_dist: int,
):
    sector_length_after_zcxy = torch.zeros_like(match_offsets_inv_zcxy)
    num_sections = match_offsets_inv_zcxy.shape[0]
    for i in range(num_sections - 1, -1, -1):
        for offset in range(1, max_dist + 1):
            j = i + offset
            if j >= num_sections:
                continue

            this_pairwise_field = pairwise_fields_zcxy[str(-offset)][j : j + 1]

            this_offset_sector_length = this_pairwise_field.sample(  # type: ignore
                sector_length_after_zcxy[j].float(), mode="nearest"
            ).int()
            this_offset_sector_length[match_offsets_inv_zcxy[i] != offset] = 0
            this_offset_sector_length[match_offsets_inv_zcxy[i] == offset] += 1

            sector_length_after_zcxy[i] = torch.max(
                sector_length_after_zcxy[i], this_offset_sector_length
            )
    return sector_length_after_zcxy
