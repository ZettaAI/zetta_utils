from __future__ import annotations

from typing import Sequence

import attrs
import torch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.index import VolumetricIndex
from zetta_utils.layer.volumetric.layer import VolumetricLayer


@builder.register("AdjustAffinitiesOp")
@mazepa.taskable_operation_cls
@attrs.frozen
class AdjustAffinitiesOp:
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_input_resolution(  # pylint: disable=no-self-use
        self, dst_resolution: Vec3D[float]
    ) -> Vec3D[float]:
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> AdjustAffinitiesOp:
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        aff_layer: VolumetricLayer,
        aff_backup_layer: VolumetricLayer,
        blackout_mask_layer: VolumetricLayer,
        snap_mask_layer: VolumetricLayer,
        threshold_mask_layer: VolumetricLayer,
        threshold_value: float = 0.85,
        fill_value: float = 0,
        rework_mode: bool = False,
        rework_mask_layer: VolumetricLayer | None = None,
    ) -> None:
        """
        Adjust affinities using masks, making sure to backup any adjusted affinities.

        :param idx: VolumetricIndex for the operation
        :param src: layer with affinities
        :param dst: layer where adjusted affinities will be written sparsely
        :param aff_backup_layer: layer where original affinities will be stored
            if chunk is adjusted
        :param aff_backup_layer: layer where original affinities will be stored
            if chunk is adjusted
        :param blackout_mask_layer: one-hot mask to black out all affinities
        :param snap_mask_layer: one-hot mask for which segments cannot span its
            boundary ("snaps" objects at its boundary)
        :param threshold_mask_layer: one-hot mask which will adjust affinities below
            `threshold_value`
        :param threshold_value: value for thresholding
        :param fill_value: value to set the adjusted affinities
        :param rework_mode: bool to use rework_mask_layer; requires `rework_mask_layer`
        :param rework_mask_layer: one-hot mask of locations that need to be reworked
        """
        idx_padded = idx.padded(self.crop_pad)
        if rework_mode:
            if rework_mask_layer is None:
                raise ValueError("`rework_mode` requires `rework_mask_layer` to be given.")
            rework_mask = rework_mask_layer[idx_padded]
            if (rework_mask != 0).sum().item() == 0:
                return
        blackout_mask = blackout_mask_layer[idx_padded]
        snap_mask = snap_mask_layer[idx_padded]
        threshold_mask = threshold_mask_layer[idx_padded]
        blackout_mask_has_data = (blackout_mask != 0).sum().item() > 0
        snap_mask_has_data = (snap_mask != 0).sum().item() > 0
        threshold_mask_has_data = (threshold_mask != 0).sum().item() > 0
        any_mask_has_data = any(
            [blackout_mask_has_data, snap_mask_has_data, threshold_mask_has_data]
        )
        if any_mask_has_data:
            aff = aff_layer[idx_padded]
            # Check the backup, in case we've run this operation before.
            # Otherwise, we may overwrite any previous backup.
            # This relies on the rest of the adjustment functions being idempotent.
            # This assumption may not be true if a restarted operation changes
            # any of the parameters (e.g. later runs lower the threshold_value).
            aff_backup = aff_backup_layer[idx_padded]
            # A 0-valued affinity indicates there may have been a backup
            aff[aff == 0] = aff_backup[aff == 0]
            if aff.sum() > 0:
                aff_backup = aff.clone()
                if snap_mask_has_data:
                    aff = adjust_affinities_across_mask_boundary(
                        src=aff, mask=snap_mask, fill_value=fill_value
                    )
                if threshold_mask_has_data:
                    aff = adjust_thresholded_affinities_in_mask(
                        src=aff,
                        mask=threshold_mask,
                        threshold_value=threshold_value,
                        fill_value=fill_value,
                    )
                if blackout_mask_has_data:
                    blackout_mask = torch.cat([blackout_mask, blackout_mask, blackout_mask], dim=0)
                    aff[blackout_mask] = fill_value
                aff_adjusted_mask = aff != aff_backup
                # store only affinities that are different
                # assumes that a 0 affinity is never changed
                aff_backup[~aff_adjusted_mask] = 0
                aff_backup_layer[idx] = tensor_ops.crop(aff_backup, self.crop_pad)
                dst[idx] = tensor_ops.crop(aff, self.crop_pad)


@builder.register("adjust_affinities_across_mask_boundary")
def adjust_affinities_across_mask_boundary(
    src: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0,
) -> torch.Tensor:
    """
    Adjust affinities that span masked and unmasked voxel pairs.

    Note: At each voxel, we store the affinities to neighboring voxels in
    the negative cardinal directions. Meaning that location (1, 1, 1) in
    the affinity map contains the following affinities:
    (0, 1, 1) - (1, 1, 1)
    (1, 0, 1) - (1, 1, 1)
    (1, 1, 0) - (1, 1, 1)

    :param src: input Tensor
    :param mask: one-hot mask matching dimensions of src
    :param fill_value: value to set the adjusted affinities
    """
    result = src
    mask_compare = mask[0, 1:, :, :] != mask[0, :-1, :, :]
    aff_mask = torch.nn.functional.pad(mask_compare, (0, 0, 0, 0, 1, 0))
    result[0, aff_mask] = fill_value
    mask_compare = mask[0, :, 1:, :] != mask[0, :, :-1, :]
    aff_mask = torch.nn.functional.pad(mask_compare, (0, 0, 1, 0, 0, 0))
    result[1, aff_mask] = fill_value
    mask_compare = mask[0, :, :, 1:] != mask[0, :, :, :-1]
    aff_mask = torch.nn.functional.pad(mask_compare, (1, 0, 0, 0, 0, 0))
    result[2, aff_mask] = fill_value
    result = result.to(src.dtype)
    return result


@builder.register("adjust_thresholded_affinities_in_mask")
def adjust_thresholded_affinities_in_mask(
    src: torch.Tensor,
    mask: torch.Tensor,
    threshold_value: float = 0.85,
    fill_value: float = 0,
) -> torch.Tensor:
    """
    Adjust affinities below `threshold_value` that are stored at masked voxels.

    :param src: input Tensor
    :param mask: one-hot mask matching single-channel dimensions of src
    :param threshold_value: affinities below this value will be set to `fill_value`
    :param fill_value: value to set the adjusted affinities
    """
    result = src
    aff_mask = torch.cat([mask, mask, mask], dim=0)
    threshold_mask = torch.logical_and(src < threshold_value, aff_mask)
    result[threshold_mask] = fill_value
    return result
