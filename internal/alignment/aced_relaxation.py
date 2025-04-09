# pylint: disable=too-many-locals
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence

import attrs
import torch
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, log
from zetta_utils.layer import volumetric
from zetta_utils.tensor_ops import common, convert
from zetta_utils.tensor_typing import TensorTypeVar

from .field import get_rigidity_map_zcxy

logger = log.get_logger("zetta_utils")


def compute_aced_loss_new(
    pfields_raw: Dict[int, torch.Tensor],
    afields: List[torch.Tensor],
    match_offsets: List[torch.Tensor],
    rigidity_weight: float,
    rigidity_scales: tuple[int, ...],
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

    with torch.no_grad():
        rigidity_masks_warped = afields_cat(rigidity_masks.float())  # type: ignore
        rigidity_masks_warped[
            rigidity_masks_warped < min_rigidity_multiplier
        ] = min_rigidity_multiplier

    for scale in rigidity_scales:
        afields_scaled = (
            torch.nn.functional.interpolate(
                afields_cat.pixels(),  # type: ignore
                scale_factor=(1 / scale, 1 / scale),
                mode="bilinear",
            )
            / scale
        )
        intra_loss_map = get_rigidity_map_zcxy(afields_scaled)

        with torch.no_grad():
            rigidity_masks_warped_scaled = torch.nn.functional.interpolate(
                rigidity_masks_warped, scale_factor=(1 / scale, 1 / scale), mode="bilinear"
            )
        intra_loss += (
            (intra_loss_map * rigidity_weight * rigidity_masks_warped_scaled.squeeze()).sum()
            * scale
            * scale
        )

    loss = inter_loss / (afields[0].shape[-1] * afields[0].shape[-2]) + intra_loss / (
        afields[0].shape[-1] * afields[0].shape[-1] * len(rigidity_scales)
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


def _optimize(
    pfields_raw: Dict[int, torchfields.Field],
    afields: List[torchfields.Field],
    match_offsets_zcxy: torch.Tensor,
    rigidity_masks_zcxy: torch.Tensor,
    rigidity_weight: float,
    rigidity_scales: Sequence[int],
    max_dist: int,
    min_rigidity_multiplier: float,
    num_iter: int,
    lr: float,
    fix: Literal["first", "last", "both"] | None,
    grad_clip: float | None,
):
    num_sections = match_offsets_zcxy.shape[0]
    opt_range = _get_opt_range(fix=fix, num_sections=num_sections)
    for i in opt_range:
        afields[i].requires_grad = True

    optimizer = torch.optim.Adam(
        [afields[i] for i in opt_range],
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100
    )

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
        for i in range(num_iter):
            loss_new = compute_aced_loss_new(
                pfields_raw=pfields_raw,
                afields=afields,
                rigidity_masks=rigidity_masks_zcxy,
                match_offsets=[match_offsets_zcxy[i] for i in range(num_sections)],
                rigidity_weight=rigidity_weight,
                rigidity_scales=tuple(rigidity_scales),
                max_dist=max_dist,
                min_rigidity_multiplier=min_rigidity_multiplier,
            )
            loss = loss_new
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(afields, max_norm=grad_clip)
            optimizer.step()
            scheduler.step(loss)
            if i % 100 == 0:
                last_lr = scheduler.get_last_lr()[0]
                logger.info(f"Iter {i} loss: {loss} | lr: {last_lr}")
                if last_lr < 1e-5:
                    logger.info(f"No notable improvement - stopping at iter {i} loss: {loss}")
                    break

    return afields


@builder.register("perform_aced_relaxation")
def perform_aced_relaxation(  # pylint: disable=too-many-branches
    match_offsets: TensorTypeVar,
    pfields: dict[str, TensorTypeVar],
    rigidity_masks: TensorTypeVar | None = None,
    first_section_fix_field: TensorTypeVar | None = None,
    last_section_fix_field: TensorTypeVar | None = None,
    initial_field: TensorTypeVar | None = None,
    min_rigidity_multiplier: float = 0.0,
    num_iter=100,
    lr=0.3,
    rigidity_weight=10.0,
    rigidity_scales: Sequence[int] = (1,),
    fix: Optional[Literal["first", "last", "both"]] = None,
    max_dist: int = 2,
    grad_clip: float | None = None,
) -> volumetric.VolumetricLayerDType:
    assert "-1" in pfields

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    max_displacement = max(common.abs(field).max().item() for field in pfields.values())

    if not match_offsets.any() or max_displacement < 0.01:
        if initial_field is None:
            return volumetric.to_vol_layer_dtype(torch.zeros(pfields["-1"].shape))
        return volumetric.to_vol_layer_dtype(initial_field)

    match_offsets_zcxy = convert.to_torch(
        common.rearrange(match_offsets, "C X Y Z -> Z C X Y"), device=device
    )

    if rigidity_masks is not None:
        rigidity_masks_zcxy = convert.to_torch(
            common.rearrange(rigidity_masks, "C X Y Z -> Z C X Y"), device=device
        )
    else:
        rigidity_masks_zcxy = torch.ones_like(match_offsets_zcxy)

    num_sections = match_offsets_zcxy.shape[0]
    assert num_sections > 1, "Can't relax blocks with just one section"

    pfields_raw: Dict[int, torchfields.Field] = {}

    for offset_str, field in pfields.items():
        offset = -int(offset_str)
        pfields_raw[offset] = (
            convert.to_torch(common.rearrange(field, "C X Y Z -> Z C X Y"), device=device)
            .field_()  # type: ignore
            .from_pixels()
        )

    if initial_field is None:
        afields = [
            torch.zeros(
                (1, 2, match_offsets_zcxy.shape[2], match_offsets_zcxy.shape[3]), device=device
            )
            .field_()  # type: ignore
            .from_pixels()
            for _ in range(num_sections)
        ]
    else:
        afields = [
            convert.to_torch(initial_field[..., z], device=device)
            .unsqueeze(0)
            .field_()  # type: ignore
            .from_pixels()
            for z in range(num_sections)
        ]

    if first_section_fix_field is not None:
        assert fix in ["first", "both"]

        first_section_fix_field_zcxy: torchfields.Field = (
            convert.to_torch(
                common.rearrange(first_section_fix_field, "C X Y Z -> Z C X Y"), device=device
            )
            .field_()  # type: ignore
            .from_pixels()
        )

        afields[0] = first_section_fix_field_zcxy

    if last_section_fix_field is not None:
        assert fix in ["last", "both"]

        last_section_fix_field_zcxy = (
            convert.to_torch(
                common.rearrange(last_section_fix_field, "C X Y Z -> Z C X Y"),  # type: ignore
                device=device,
            )
            .field_()
            .from_pixels()
        )

        afields[-1] = last_section_fix_field_zcxy

    afields = _optimize(
        pfields_raw=pfields_raw,
        afields=afields,
        match_offsets_zcxy=match_offsets_zcxy,
        rigidity_masks_zcxy=rigidity_masks_zcxy,
        rigidity_weight=rigidity_weight,
        rigidity_scales=rigidity_scales,
        max_dist=max_dist,
        min_rigidity_multiplier=min_rigidity_multiplier,
        num_iter=num_iter,
        lr=lr,
        fix=fix,
        grad_clip=grad_clip,
    )

    result_xy = torch.cat(afields, 0).pixels()  # type: ignore
    result = volumetric.to_vol_layer_dtype(
        common.rearrange(result_xy, "Z C X Y -> C X Y Z"),
    )
    return result


def get_aced_match_offsets_naive(
    non_tissue: TensorTypeVar,
    misalignment_mask_zm1: TensorTypeVar,
    misalignment_mask_zm2: Optional[TensorTypeVar] = None,
    misalignment_mask_zm3: Optional[TensorTypeVar] = None,
) -> volumetric.VolumetricLayerDType:

    match_offsets = torch.ones_like(convert.to_torch(non_tissue), dtype=torch.int32) * -1
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
    result = volumetric.to_vol_layer_dtype(match_offsets.byte())
    return result


def get_aced_match_offsets(
    tissue_mask: TensorTypeVar,
    misalignment_masks: dict[str, TensorTypeVar],
    pairwise_fields: dict[str, TensorTypeVar],
    pairwise_fields_inv: dict[str, TensorTypeVar],
    max_dist: int,
) -> dict[str, volumetric.VolumetricLayerDType]:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
        tissue_mask_zcxy = convert.to_torch(
            common.rearrange(tissue_mask, "1 X Y Z -> Z 1 X Y"), device=device
        )
        misalignment_masks_zcxy = {
            k: convert.to_torch(common.rearrange(v, "1 X Y Z -> Z 1 X Y"), device=device)
            for k, v in misalignment_masks.items()
        }
        pairwise_fields_zcxy = {
            k: convert.to_torch(common.rearrange(v, "C X Y Z -> Z C X Y"), device=device)
            .field_()  # type: ignore
            .from_pixels()
            for k, v in pairwise_fields.items()
        }
        pairwise_fields_inv_zcxy = {
            k: convert.to_torch(common.rearrange(v, "C X Y Z -> Z C X Y"), device=device)
            .field_()  # type: ignore
            .from_pixels()
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
    result = {
        k: volumetric.to_vol_layer_dtype(common.rearrange(v, "Z C X Y -> C X Y Z"))
        for k, v in result.items()
    }
    return result


@attrs.mutable
class _FwdPassOutcome:
    sector_length_before_zcxy: torch.Tensor
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

    sector_length_before_zcxy = torch.zeros_like(tissue_mask_zcxy).int()
    match_offsets_zcxy = torch.zeros_like(tissue_mask_zcxy).int()
    match_offsets_inv_zcxy = torch.zeros_like(tissue_mask_zcxy).int()

    for curr_z in range(1, num_sections):
        offset_scores = torch.zeros(
            (max_dist, 1, tissue_mask_zcxy.shape[-2], tissue_mask_zcxy.shape[-1]),
            dtype=torch.int32,
            device=tissue_mask_zcxy.device,
        )

        offset_sector_lengths = torch.zeros(
            (max_dist, 1, tissue_mask_zcxy.shape[-2], tissue_mask_zcxy.shape[-1]),
            dtype=torch.int32,
            device=tissue_mask_zcxy.device,
        )

        for offset in range(1, max_dist + 1):
            curr_z_minus_offset = curr_z - offset
            if curr_z_minus_offset < 0:
                break

            # Field that aligns z-offset to z
            this_pairwise_field_inv = pairwise_fields_inv_zcxy[str(-offset)][curr_z : curr_z + 1]

            # Warp tissue mask at z-offset to align with z
            tgt_tissue_mask = this_pairwise_field_inv.sample(  # type: ignore
                tissue_mask_zcxy[curr_z_minus_offset].float(),
                mode="nearest",
            ).int()

            # Intersection of tissue masks z and warped z-offset
            this_tissue_mask = tissue_mask_zcxy[curr_z] * tgt_tissue_mask

            # Misalignment mask between z and warped z-offset
            this_misalignment_mask = misalignment_masks_zcxy[str(-offset)][curr_z]

            # Calculate would-be sector lengths for this match offset, "+ offset" because
            # we want to maximize real length of aligned stretch, not only chain link count
            offset_sector_lengths[offset - 1] = (
                this_pairwise_field_inv.sample(  # type: ignore
                    sector_length_before_zcxy[curr_z_minus_offset].float(),
                    mode="nearest",
                ).int()
                + offset
            )

            # Deprioritize matches that are non-tissue
            offset_sector_lengths[offset - 1][this_tissue_mask == 0] = 0

            # Deprioritize matches that are not aligned
            offset_sector_lengths[offset - 1][this_misalignment_mask != 0] = 0
            assert offset_sector_lengths[offset - 1].max() < 2 ** 21

            offset_scores[offset - 1] = (this_tissue_mask != 0) * 2 ** 30
            offset_scores[offset - 1] += (
                misalignment_masks_zcxy[str(-offset)][curr_z] == 0
            ) * 2 ** 29
            # Prioritize longer match chains
            offset_scores[offset - 1] += offset_sector_lengths[offset - 1] * 2 ** 8
            # if equal, pick nearest match
            offset_scores[offset - 1] += max_dist - offset

        chosen_offset_scores, chosen_offsets = offset_scores.max(0)
        passable_choices = chosen_offset_scores >= (2 ** 30 + 2 ** 29)
        match_offsets_zcxy[curr_z][passable_choices] = chosen_offsets[passable_choices].int() + 1
        # match_offsets_zcxy[curr_z] = this_tissue_mask

        # sector_length_before_zcxy[curr_z] = offset_sector_lengths[chosen_offsets]
        # TODO: how do vectorize this?
        for choice in range(0, max_dist):
            this_match_locations = chosen_offsets == choice
            sector_length_before_zcxy[curr_z][this_match_locations] = offset_sector_lengths[
                choice
            ][this_match_locations]

        for offset in range(1, max_dist + 1):
            curr_z_minus_offset = curr_z - offset
            this_offset_matches = match_offsets_zcxy[curr_z] == offset

            # Discard non-aligned matches for bwd pass
            this_offset_matches[chosen_offset_scores < (2 ** 30 + 2 ** 29)] = 0
            if this_offset_matches.sum() > 0:
                this_inv_field = pairwise_fields_zcxy[str(-offset)][curr_z : curr_z + 1]
                this_offset_matches_inv = this_inv_field.sample(  # type: ignore
                    this_offset_matches.float(), mode="nearest"
                ).int()
                this_offset_matches_inv[tissue_mask_zcxy[curr_z_minus_offset] == 0] = 0
                match_offsets_inv_zcxy[curr_z_minus_offset][this_offset_matches_inv != 0] = offset

    return _FwdPassOutcome(
        sector_length_before_zcxy=sector_length_before_zcxy,
        match_offsets_zcxy=match_offsets_zcxy,
        match_offsets_inv_zcxy=match_offsets_inv_zcxy,
    )


def _get_masks(
    sector_length_before_zcxy: torch.Tensor,
    sector_length_after_zcxy: torch.Tensor,
    # pairwise_fields_inv_zcxy: dict[str, TensorTypeVar],
    # match_offsets_zcxy: TensorTypeVar,
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
    for curr_z in range(num_sections - 1, -1, -1):
        for offset in range(1, max_dist + 1):
            curr_z_plus_offset = curr_z + offset
            if curr_z_plus_offset >= num_sections:
                continue

            this_pairwise_field = pairwise_fields_zcxy[str(-offset)][
                curr_z_plus_offset : curr_z_plus_offset + 1
            ]

            this_offset_sector_length = this_pairwise_field.sample(  # type: ignore
                sector_length_after_zcxy[curr_z_plus_offset].float(), mode="nearest"
            ).int()
            this_offset_sector_length[match_offsets_inv_zcxy[curr_z] != offset] = 0
            this_offset_sector_length[match_offsets_inv_zcxy[curr_z] == offset] += offset

            sector_length_after_zcxy[curr_z] = torch.max(
                sector_length_after_zcxy[curr_z], this_offset_sector_length
            )
    return sector_length_after_zcxy
