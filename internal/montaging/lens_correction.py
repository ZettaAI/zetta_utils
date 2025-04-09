from __future__ import annotations

from typing import Sequence

import attrs
import torch
import torchfields
import torchvision

from zetta_utils import builder, log, mazepa, tensor_ops
from zetta_utils.geometry import Vec3D
from zetta_utils.geometry.bbox import BBox3D
from zetta_utils.layer.db_layer import DBLayer
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol import build_cv_layer
from zetta_utils.tensor_ops import convert

from .registry import dict_to_chunks, zs_to_tiles

logger = log.get_logger("zetta_utils")


@builder.register("LensCorrectionModel")
@attrs.frozen
class LensCorrectionModel:
    """
    Wrapper for storing and applying lens correction models for tiles at different resolutions,
    with automatic interpolation.

    Note that by convention, lens correction models are stored with (0, 0, 0)
    voxel offset, in a single backend chunk with the size given by
    `_get_tile_size_with_pad_in_res()`.

    :param path: CloudVolume path where the model is stored.
    :param full_res: Resolution that the tiles are in, in XY.
    :param tile_size_full: Tile size at full resolution.
    :param model_res: Resolution that the model is written at, in XY.
    :param pad_in_model_res: Padding in each direction in XY for the lens correction
        model, at the model resolution. For instance, `pad_in_model_res` of 64 for
        a 16nm lens correction model for a 4nm res 2048x2048 tile would mean
        that the lens correction model is 640x640 (2048 * 4 / 16 + 64) in size.
    """

    path: str
    full_res: float
    tile_size_full: int
    model_res: float
    pad_in_model_res: int

    def _get_tile_size_with_pad_at_res(self, res: float | None = None) -> int:
        """Returns the expected tile size with the padding at a given resolution."""
        res_ = self.model_res if res is None else res
        return 2 * round(self.pad_in_model_res * self.model_res / res_) + round(
            self.tile_size_full * self.full_res / res_
        )

    @property
    def _layer(self) -> VolumetricLayer:
        return build_cv_layer(
            path=self.path,
            interpolation_mode="field",
            data_resolution=Vec3D(self.model_res, self.model_res, 1),
        )

    def _get_index_at_res(self, res: float) -> VolumetricIndex:
        """Returns the VolumetricIndex to use to retrieve the lens correction model at
        a given resolution.
        """
        tile_size_in_res_with_pad = self._get_tile_size_with_pad_at_res(res)
        return VolumetricIndex.from_coords(
            start_coord=Vec3D(0, 0, 0),
            end_coord=Vec3D(tile_size_in_res_with_pad, tile_size_in_res_with_pad, 1),
            resolution=Vec3D(res, res, 1),
        )

    def _get_model_at_res(self, res: float) -> torchfields.Field:
        """Returns the lens correction model at a given resolution."""
        return (
            tensor_ops.common.rearrange(
                self._layer[self._get_index_at_res(res)], pattern="C X Y Z -> Z C X Y"
            )
            .field_()  # type: ignore
            .from_pixels()
        )

    def pad_at_res(self, res: float | None = None) -> int:
        """Returns the padding for the model in each direction at a given resolution."""
        res_ = self.model_res if res is None else res
        return round(self.pad_in_model_res * self.model_res / res_)

    def apply_at_res(self, data: torch.Tensor, res: float) -> torch.Tensor:
        """Applies the model to a CXYZ Tensor representing the unpadded tile(s) at resolution
        `res`.
        Returns padded output.

        :param data: The CXYZ tensor representing the unpadded tile(s).
        :param res: Resolution that the tiles are given at.
        """
        model = self._get_model_at_res(res)

        return model(
            torchvision.transforms.Pad(self.pad_at_res(res), 0, "constant")(data.float())
        ).to(data.dtype)


# TODO: arbitrary / nonisotropic sizes
@builder.register("estimate_lens_distortion_from_registry_flow")
@mazepa.flow_schema
def estimate_lens_distortion_from_registry_flow(  # pylint:disable=too-many-locals, invalid-name
    tile_registry: DBLayer,
    field_paths: list[str],
    output_path: str,
    model_res: Sequence[float],
    pad_in_model_res: int,
    full_res: Sequence[float],
    tile_size_full: int,
    z_start: int,
    z_stop: int,
    num_tasks: int = 200,
):
    """
    Estimates the lens distortion from the fields for the tiles in the given registry
    with Z offsets in range (z_start, z_stop). Saves the final output in the `output_path`
    layer in the LensCorrectionModel convention.

    :param tile_registry: The tile registry.
    :param field_paths: Paths for the 4 fields, in 0_0, 0_1, 1_0, 1_1 order.
    :param output_path: Output location.
    :param model_res: Resolution to generate the lens distortion model. The fields MUST
        have information at this model.
    :param pad_in_model_res: How many pixels to pad the tiles by in XY, at `model_res`.
    :param full_res: The full resolution that the tiles are in.
    :param tile_size_full: The tile size in XY at full resolution.
    :param z_start: Z offset to start, inclusive.
    :param z_stop: Z offset to stop, exclusive.
    :param num_tasks: Number of tasks to generate.
    """
    model_res = Vec3D(*model_res)
    full_res = Vec3D(*full_res)
    assert model_res[2] == full_res[2]
    resolution_lens_models = Vec3D(model_res[0], model_res[1], 1)

    # assumes isotropic resolution
    tile_size_in_res_with_pad = 2 * pad_in_model_res + round(
        tile_size_full * full_res[0] / model_res[0]
    )
    output_layer = build_cv_layer(
        path=output_path,
        info_type="image",
        info_data_type="float32",
        info_num_channels=2,
        info_overwrite=True,
        info_scales=[resolution_lens_models],
        info_encoding="raw",
        info_chunk_size=[tile_size_in_res_with_pad, tile_size_in_res_with_pad, 1],
        info_bbox=BBox3D.from_coords(
            start_coord=[0, 0, 0],
            end_coord=[tile_size_in_res_with_pad, tile_size_in_res_with_pad, num_tasks + 1],
            resolution=resolution_lens_models,
        ),
    )

    sublayer_list = [
        build_cv_layer(
            path=field_path,
        )
        for field_path in field_paths
    ]

    all_tiles = zs_to_tiles(tile_registry, z_start, z_stop)

    sum_tasks = []

    for i, chunk in enumerate(dict_to_chunks(all_tiles, len(all_tiles) // num_tasks)):
        paths = list(chunk.keys())
        xs = [chunk[path]["x_index"] for path in paths]
        ys = [chunk[path]["y_index"] for path in paths]
        offsets_full_res: list[Vec3D] = [
            Vec3D(chunk[path]["x_offset"], chunk[path]["y_offset"], chunk[path]["z_offset"])
            for path in paths
        ]
        sublayers = [sublayer_list[2 * (x % 2) + (y % 2)] for x, y in zip(xs, ys)]

        sum_tasks.append(
            download_and_sum_tile_fields.make_task(
                sublayers,
                offsets_full_res,
                model_res,
                pad_in_model_res,
                tile_size_full,
                full_res,
                output_layer,
                i + 1,
            )
        )

    yield sum_tasks
    yield mazepa.Dependency()
    for task in sum_tasks:
        assert task.outcome is not None
        assert task.outcome.return_value is not None

    cts = sum(task.outcome.return_value for task in sum_tasks)  # type: ignore

    # CYXZ
    idx_to_fetch = VolumetricIndex.from_coords(
        start_coord=Vec3D(0, 0, 1),
        end_coord=round(
            Vec3D(tile_size_full, tile_size_full, num_tasks + 1) * full_res / model_res
        ),
        resolution=model_res,
    ).translated_end(2 * Vec3D(pad_in_model_res, pad_in_model_res, 0))
    res = torch.sum(convert.to_torch(output_layer[idx_to_fetch]), 3).unsqueeze(-1) / cts
    idx = VolumetricIndex.from_coords(
        start_coord=Vec3D(0, 0, 0),
        end_coord=round(
            Vec3D(tile_size_full, tile_size_full, 1) * full_res / resolution_lens_models
        ),
        resolution=resolution_lens_models,
    ).translated_end(2 * Vec3D(pad_in_model_res, pad_in_model_res, 0))
    output_layer[idx] = res

    logger.info(f"Estimated lens distortion based on {cts} total tiles.")


@mazepa.taskable_operation
def download_and_sum_tile_fields(
    sublayers: list[VolumetricLayer],
    offsets_full_res: list[Vec3D],
    model_res: Vec3D[float],
    pad_in_model_res: int,
    tile_size_full: int,
    full_res: Vec3D[float],
    output_layer: VolumetricLayer,
    output_z_ind: int,
) -> int:
    """
    Downloads the fields corresponding to tiles at ``x_offset``, ``y_offset``, ``z_offset``
    for the corresponding ``sublayer``. Offsets are assumed to be at full resolution.

    Writes the sum to ``output_layer`` at the ``output_z_ind`` for averaging.

    :param sublayers: Sublayers where each of the tiles stored.
    :param offsets_full_res: X, Y, Z offsets of each of the tiles, given at full resolution.
    :param full_res: Resolution that the tiles are in, in XY.
    :param tile_size_full: Tile size at full resolution.
    :param model_res: Resolution that the model will be read / written at, in XY.
    :param pad_in_model_res: Padding in each direction in XY for the lens correction
        model, at the model resolution. For instance, `pad_in_model_res` of 64 for
        a 16nm lens correction model for a 4nm res 2048x2048 tile would mean
        that the lens correction model is 640x640 (2048 * 4 / 16 + 64) in size.
    :param output_layer: Layer to write the output to.
    :param output_z_ind: Z index to write the output to.
    """
    count = 0
    resolution_lens_models = Vec3D(model_res[0], model_res[1], full_res[2])
    for sublayer, offset_full_res in zip(sublayers, offsets_full_res):
        idx = VolumetricIndex.from_coords(
            start_coord=round(offset_full_res * full_res / resolution_lens_models),
            end_coord=round(
                (offset_full_res + Vec3D(tile_size_full, tile_size_full, 1))
                * full_res
                / resolution_lens_models
            ),
            resolution=model_res,
        ).padded(Vec3D(pad_in_model_res, pad_in_model_res, 0))
        sublayer_data = convert.to_torch(sublayer[idx])
        if torch.count_nonzero(sublayer_data).item() != 0:
            if count == 0:
                tiles_sum = sublayer_data
            else:
                tiles_sum += sublayer_data
            count += 1
    idx = VolumetricIndex.from_coords(
        start_coord=Vec3D(0, 0, output_z_ind),
        end_coord=round(
            Vec3D(tile_size_full, tile_size_full, output_z_ind + 1)
            * full_res
            / resolution_lens_models
        ),
        resolution=resolution_lens_models,
    ).translated_end(2 * Vec3D(pad_in_model_res, pad_in_model_res, 0))
    if count != 0:
        output_layer[idx] = tiles_sum

    return count
