from __future__ import annotations

from typing import Sequence

import attrs
import torch
import torchfields
import torchvision

from zetta_utils import builder, log, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricLayer,
    VolumetricLayerSet,
)
from zetta_utils.mazepa import semaphore
from zetta_utils.tensor_ops.common import interpolate

from ..alignment.field import get_rigidity_map_zcxy

logger = log.get_logger("zetta_utils")


def compute_dumpster_loss(
    match_fields: list[torchfields.Field],
    afields: list[torchfields.Field],
    fields_layout: tuple[tuple[int, int], ...],
    rigidity_weight: float,
    rigidity_masks: torch.Tensor,
    rigidity_scales: tuple[int, ...],
) -> torch.Tensor:
    intra_loss = 0
    inter_loss = 0

    afields_cat = torch.cat(afields)
    with torch.no_grad():
        rigidity_masks_warped = afields_cat(rigidity_masks)  # type: ignore

    for scale in rigidity_scales:
        afields_scaled = (
            torch.nn.functional.interpolate(
                afields_cat.pixels(), scale_factor=(1 / scale, 1 / scale), mode="bilinear"  # type: ignore # pylint: disable = line-too-long
            )
            / scale
        )
        intra_loss_map = get_rigidity_map_zcxy(afields_scaled)

        with torch.no_grad():
            rigidity_masks_warped_scaled = torch.nn.functional.interpolate(
                rigidity_masks_warped, scale_factor=(1 / scale, 1 / scale), mode="bilinear"
            )
        intra_loss += (
            intra_loss_map * rigidity_weight * rigidity_masks_warped_scaled.squeeze()
        ).sum() * scale

    inter_loss = 0

    for src, tgt in fields_layout:  # TODO
        inter_loss_map = ((afields[tgt])(match_fields[src]) - afields[src]).pixels()
        inter_loss_map_mask = rigidity_masks_warped[src] * rigidity_masks_warped[tgt]
        this_inter_loss = ((inter_loss_map ** 2).sum(1) * inter_loss_map_mask).sum()
        inter_loss += this_inter_loss

    loss = inter_loss / (
        afields[0].shape[-1] * afields[0].shape[-2] * len(fields_layout)
    ) + intra_loss / (afields[0].shape[-1] * afields[0].shape[-1] * len(rigidity_scales))
    return loss


def perform_relaxation(  # pylint: disable=too-many-branches
    match_fields: list[torchfields.Field],
    afields: list[torchfields.Field],
    lr,
    rigidity_scales: tuple[int, ...],
    fields_layout: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0)),
    rigidity_masks: torch.Tensor | None = None,
    num_iter=500,
    rigidity_weight=1.0,
    rigidity_mask_gaussian_blur_sigma=25.0,
    grad_clip: float = 0.05,
) -> list[torchfields.Field] | None:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    afields = [afield.from_pixels().cuda() for afield in afields]
    max_displacement = max(field.abs().max().item() for field in match_fields)

    if max_displacement < 0.01:
        return None

    num_sections = 4  # TODO
    assert num_sections > 1, "Can't relax blocks with just one section"

    if rigidity_masks is not None:
        rigidity_masks_zcxy = torchvision.transforms.GaussianBlur(
            rigidity_mask_gaussian_blur_sigma * 4 + 1, sigma=rigidity_mask_gaussian_blur_sigma
        )(rigidity_masks).to(device)
    else:
        rigidity_masks_zcxy = torch.ones(
            (num_sections, 1, match_fields[0].shape[2], match_fields[0].shape[3])
        ).to(device)

    match_fields = [field.to(device).from_pixels() for field in match_fields]

    for afield in afields:
        afield.requires_grad = True

    optimizer = torch.optim.Adam(
        afields,
        lr=lr,
    )

    with torchfields.set_identity_mapping_cache(True, clear_cache=True):
        for i in range(num_iter):
            loss_new = compute_dumpster_loss(
                match_fields,
                afields=afields,
                rigidity_masks=rigidity_masks_zcxy,
                fields_layout=fields_layout,
                rigidity_weight=rigidity_weight,
                rigidity_scales=rigidity_scales,
            )
            loss = loss_new
            if i % 100 == 0:
                logger.info(f"Iter {i} loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(afields, max_norm=grad_clip)
            optimizer.step()

    for afield in afields:
        afield.requires_grad = False
    return [afield.cpu().pixels() for afield in afields]


@builder.register("MontagingRelaxOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class MontagingRelaxOperation:  # pylint: disable = no-self-use
    crop_pad: Sequence[int] = (0, 0, 0)

    def get_operation_name(self):
        return "MontagingRelaxOperation"

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution  # TODO add support for data res

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> MontagingRelaxOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayerSet,
        fields: list[VolumetricLayer],  # TODO turn into Layerset
        srcs: list[VolumetricLayer],  # TODO turn into Layerset
        num_iter=500,
        lr=0.001,
        rigidity_weight=1.0,
        src_mask_gaussian_blur_sigma: int = 25,
        rigidity_scales: Sequence[int] = (1,),
        scales_to_relax: Sequence[int] = (1,),
    ):
        idx_padded = idx.padded(self.crop_pad)
        idx_padded.resolution = self.get_input_resolution(idx_padded.resolution)

        with semaphore("read"):
            fields_data = [
                tensor_ops.convert.to_torch(
                    tensor_ops.common.rearrange(field[idx_padded], pattern="C X Y Z -> Z C X Y")
                ).field_()  # type: ignore
                for field in fields
            ]

            srcs_data = torch.cat(
                [
                    tensor_ops.common.rearrange(
                        tensor_ops.convert.to_torch(
                            tensor_ops.common.compare(src[idx_padded], "neq", 0)
                        ).float(),
                        pattern="C X Y Z -> Z C X Y",
                    )
                    for src in srcs
                ]
            )

        with semaphore("cuda"):
            afields_full = [
                torch.zeros(
                    (1, 2, fields_data[0].shape[2], fields_data[0].shape[3])
                ).field()  # type: ignore
                for _ in range(len(fields_data))
            ]

            for scale in scales_to_relax:
                afields = perform_relaxation(
                    [
                        interpolate(
                            field, scale_factor=1 / scale, mode="field", unsqueeze_input_to=4
                        )
                        for field in fields_data
                    ],
                    [
                        interpolate(
                            afield, scale_factor=1 / scale, mode="field", unsqueeze_input_to=4
                        )
                        for afield in afields_full
                    ],
                    rigidity_masks=interpolate(srcs_data, scale_factor=1 / scale),
                    rigidity_mask_gaussian_blur_sigma=src_mask_gaussian_blur_sigma,
                    num_iter=num_iter,
                    lr=lr,
                    rigidity_weight=rigidity_weight,
                    rigidity_scales=tuple(rigidity_scales),
                )
                if afields is None:
                    return
                afields_full = [
                    interpolate(field, scale_factor=scale, mode="field", unsqueeze_input_to=4)
                    for field in afields
                ]
            new_fields = {
                str(key): tensor_ops.common.crop(
                    tensor_ops.common.rearrange(field, pattern="Z C X Y -> C X Y Z"),
                    self.crop_pad,
                )
                for key, field in enumerate(afields_full)
            }
        with semaphore("write"):
            dst[idx] = new_fields  # TODO CROP
