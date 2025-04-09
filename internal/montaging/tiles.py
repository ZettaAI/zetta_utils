# pylint: disable = invalid-name
from __future__ import annotations

import os
from typing import Sequence

import attrs
import fsspec
import numpy as np
import torch
from PIL import Image

from zetta_utils import builder, log, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.db_layer import DBLayer
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricLayer,
    VolumetricLayerSet,
)
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.mazepa import semaphore
from zetta_utils.tensor_ops.multitensor import compute_pixel_error, erode_combine

from .lens_correction import LensCorrectionModel
from .registry import dict_to_chunks, zs_to_tiles

logger = log.get_logger("zetta_utils")


def open_image_from_gcs(
    path: str,
    ds_factor: int | None = None,
    crop: int = 0,
    lens_correction_model: LensCorrectionModel | None = None,
):
    """
    Opens an image from GCS as a CXYZ tensor, possibly with crop, downsampling, or lens correction.
    If lens correction is used, the image will be padded to the size of the lens correction model
    (scaled to the downsampling factor).

    :param path: Full path to the image.
    :param ds_factor: Factor to downsample by in XY.
    :param crop: Pixels to crop in XY before downsampling. Cannot be used with
        lens corretion models.
    :param lens_correction_model: Lens correction model to apply when opening, if any.
    :param tile_full_res: Nominal resolution of the tile before downsampling. Only needed if
        using lens correction.
    """
    fs = fsspec.filesystem("gs")
    img = np.array(Image.open(fs.open(os.path.join(path))))
    # TODO: ONLY FOR BMP?
    img_tp = torch.swapaxes(torch.from_numpy(img), 0, 1)
    img_tp = tensor_ops.common.crop(img_tp, (crop, crop))
    if ds_factor is not None:
        img_tp = tensor_ops.interpolate(img_tp, scale_factor=1 / ds_factor, unsqueeze_input_to=4)
    else:
        img_tp = img_tp.unsqueeze(0).unsqueeze(-1)

    if lens_correction_model is not None:
        assert crop == 0
        tile_res = (
            lens_correction_model.full_res
            if ds_factor is None
            else lens_correction_model.full_res * ds_factor
        )
        img_tp = lens_correction_model.apply_at_res(img_tp, tile_res)
    return img_tp


@builder.register("ingest_from_registry_flow")
@mazepa.flow_schema
def ingest_from_registry_flow(
    tile_registry: DBLayer,
    info_template_path: str,
    base_path: str,
    crop: int,
    resolution: Sequence[float],
    z_start: int,
    z_stop: int,
    num_tasks: int = 2000,
    lens_correction_model: LensCorrectionModel | None = None,
):
    """
    Writes all files in the given registry with Z offsets in range (z_start, z_stop)
    to the given base path in the lattice format.

    :param tile_registry: The tile registry.
    :param info_template_path: Path for the info template.
    :param base_path: Path for the base path; tiles will be written to
        `($BASE_PATH)/0_0`,`($BASE_PATH)/0_1`, `($BASE_PATH)/1_0`, and
        `($BASE_PATH)/1_1`.
    :param crop: Pixels to crop in each dimension; useful for discarding edges with
        heavy distortion for rendering.
    :param resolution: The resolution that the tiles are to be imported in.
    :param z_start: Z offset to start, inclusive.
    :param z_stop: Z offset to stop, exclusive.
    :param num_tasks: Number of tasks to generate.
    :param lens_correction_model: Lens correction model to apply during write.
    """

    sub_paths = [os.path.join(base_path, f"{i}_{j}") for i in range(2) for j in range(2)]
    sublayer_list = [
        build_cv_layer(
            path=sub_path,
            info_reference_path=info_template_path,
            cv_kwargs={"non_aligned_writes": True},
        )
        for sub_path in sub_paths
    ]

    all_tiles = zs_to_tiles(tile_registry, z_start, z_stop)

    upload_tasks = []

    for chunk in dict_to_chunks(all_tiles, len(all_tiles) // num_tasks):
        paths = list(chunk.keys())
        xs = [chunk[path]["x_index"] for path in paths]
        ys = [chunk[path]["y_index"] for path in paths]
        offsets_full_res: list[Vec3D] = [
            Vec3D(chunk[path]["x_offset"], chunk[path]["y_offset"], chunk[path]["z_offset"])
            for path in paths
        ]
        tile_resolutions: list[Vec3D] = [
            Vec3D(chunk[path]["x_res"], chunk[path]["y_res"], chunk[path]["z_res"])
            for path in paths
        ]
        sublayers = [sublayer_list[2 * (x % 2) + (y % 2)] for x, y in zip(xs, ys)]

        upload_tasks.append(
            upload_tiles.make_task(
                paths,
                sublayers,
                offsets_full_res,
                tile_resolutions,
                Vec3D(*resolution),
                crop,
                lens_correction_model,
            )
        )

    yield upload_tasks
    yield mazepa.Dependency()


@mazepa.taskable_operation
def upload_tiles(
    paths: list[str],
    sublayers: list[VolumetricLayer],
    offsets: list[Vec3D],
    tile_resolutions: list[Vec3D],
    resolution: Vec3D,
    crop: int,
    lens_correction_model: LensCorrectionModel | None,
) -> None:
    pad_in_res = (
        lens_correction_model.pad_at_res(resolution[0]) if lens_correction_model is not None else 0
    )
    for path, sublayer, offset, tile_resolution in zip(
        paths, sublayers, offsets, tile_resolutions
    ):
        x, y, z = round(offset * tile_resolution / resolution)
        img = open_image_from_gcs(
            path, tile_resolution[0] / resolution[0], crop, lens_correction_model
        ).to(torch.uint8)
        idx = VolumetricIndex.from_coords(
            start_coord=Vec3D(x + crop - pad_in_res, y + crop - pad_in_res, z),
            end_coord=Vec3D(
                x + crop - pad_in_res + img.shape[0], y + crop - pad_in_res + img.shape[1], z + 1
            ),
            resolution=resolution,
        )
        sublayer[idx] = img.unsqueeze(0).unsqueeze(-1)
        logger.info(
            f"Uploaded {path} to {sublayer.name} at {idx.pformat()} at resolution {resolution}."
        )


@builder.register("ComposeWithErrorsOperation")
@mazepa.taskable_operation_cls
@attrs.frozen()
class ComposeWithErrorsOperation:  # pylint:disable = no-self-use
    """
    Wrapper for storing and applying lens correction models for tiles at different resolutions,
    with automatic interpolation.
    """

    crop_pad: Sequence[int] = (0, 0, 0)

    def get_operation_name(self):
        return "ComposeWithErrorsOperation"

    def get_input_resolution(self, dst_resolution: Vec3D[float]) -> Vec3D[float]:
        return dst_resolution  # TODO add support for data res

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> ComposeWithErrorsOperation:
        return attrs.evolve(self, crop_pad=Vec3D(*self.crop_pad) + crop_pad)

    def __call__(  # pylint: disable=too-many-locals
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayerSet,
        data1: VolumetricLayer,
        data2: VolumetricLayer,
        data3: VolumetricLayer,
        data4: VolumetricLayer,
        erosion: int,
    ):
        idx_padded = idx.padded(self.crop_pad)
        idx_padded.resolution = self.get_input_resolution(idx_padded.resolution)

        with semaphore("read"):
            data1_tensor = data1[idx_padded]
            data2_tensor = data2[idx_padded]
            data3_tensor = data3[idx_padded]
            data4_tensor = data4[idx_padded]

        with semaphore("cpu"):
            output = erode_combine(
                erode_combine(data1_tensor, data2_tensor, erosion),
                erode_combine(data3_tensor, data4_tensor, erosion),
                erosion,
            )
            errors = (
                compute_pixel_error(data1_tensor, data2_tensor, erosion)
                + compute_pixel_error(data2_tensor, data3_tensor, erosion)
                + compute_pixel_error(data3_tensor, data4_tensor, erosion)
                + compute_pixel_error(data4_tensor, data1_tensor, erosion)
            )

            res = {
                "output": tensor_ops.common.crop(output, self.crop_pad),
                "errors": tensor_ops.common.crop(errors, self.crop_pad),
            }
        with semaphore("write"):
            dst[idx] = res
