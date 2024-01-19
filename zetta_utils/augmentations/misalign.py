from __future__ import annotations

import math
import random
from typing import Any, Literal, TypeVar

import attrs
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.tools_base import JointIndexDataProcessor
from zetta_utils.layer.volumetric.index import VolumetricIndex

T = TypeVar("T", torch.Tensor, dict[str, Any])


@builder.register("SlipMisalign")
@typechecked
@attrs.mutable
class MisalignProcessor(JointIndexDataProcessor[T, VolumetricIndex]):
    """
    Minimum and maximum displacement is specified in unit.
    The selected displacement will be rounded to always be divisible
    by `disp_in_unit_must_be_divisible_by`, which should correspond to
    the lowest resolution of the layers being cropped.

    Data handling is implemented through recording the fraction
    of the misalignment relative to the total bbox size.
    """

    prob: float
    disp_min_in_unit: float
    disp_max_in_unit: float
    disp_in_unit_must_be_divisible_by: float
    mode: Literal["slip", "step"] = "slip"
    keys_to_apply: list[str] | None = None

    prepared_disp_fraction: tuple[float, float] | None = attrs.field(init=False, default=None)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        if mode != "read":
            raise NotImplementedError()
        disp_x_in_unit_magn = random.uniform(self.disp_min_in_unit, self.disp_max_in_unit)
        disp_y_in_unit_magn = random.uniform(self.disp_min_in_unit, self.disp_max_in_unit)
        disp_x_in_unit_magn_rounded = math.floor(
            disp_x_in_unit_magn / self.disp_in_unit_must_be_divisible_by
        )
        disp_y_in_unit_magn_rounded = math.floor(
            disp_y_in_unit_magn / self.disp_in_unit_must_be_divisible_by
        )
        disp_x_in_unit = disp_x_in_unit_magn_rounded * random.choice([1, -1])
        disp_y_in_unit = disp_y_in_unit_magn_rounded * random.choice([1, -1])

        disp_x_in_idx_res = disp_x_in_unit / idx.resolution[0]
        disp_y_in_idx_res = disp_y_in_unit / idx.resolution[1]

        # self.prepared_disp = (disp_x, disp_y)

        start_offset = [0, 0, 0]
        end_offset = [0, 0, 0]
        if disp_x_in_idx_res > 0:
            start_offset[0] = -disp_x_in_idx_res
        else:
            end_offset[0] = -disp_x_in_idx_res
        if disp_y_in_idx_res > 0:
            start_offset[1] = -disp_y_in_idx_res
        else:
            end_offset[1] = -disp_y_in_idx_res

        idx = idx.translated_start(Vec3D[int](*start_offset)).translated_end(
            Vec3D[int](*end_offset)
        )
        self.prepared_disp_fraction = (
            disp_x_in_idx_res / idx.shape[0],
            disp_y_in_idx_res / idx.shape[0],
        )
        return idx

    def process_data(self, data: T, mode: Literal["read", "write"]) -> T:
        if mode != "read":
            raise NotImplementedError()

        if isinstance(data, torch.Tensor):
            z_size = data.shape[-1]
        else:
            if not self.keys_to_apply:
                raise ValueError(
                    "`keys_to_apply` must be a non-empty list of springs when "
                    "applying to data of type `dict`"
                )
            z_sizes = [data[k].shape[-1] for k in self.keys_to_apply]
            assert all(e == z_sizes[0] for e in z_sizes)
            z_size = z_sizes[0]

        z_chosen = random.randint(0, z_size - 1)

        if self.mode == "slip":
            z_misal_slice = slice(z_chosen, z_chosen + 1)
        else:
            z_misal_slice = slice(z_chosen, z_size)

        def _process_tensor(tensor: torch.Tensor) -> torch.Tensor:
            assert self.prepared_disp_fraction is not None
            x_offset = int(tensor.shape[-3] * self.prepared_disp_fraction[0])
            y_offset = int(tensor.shape[-2] * self.prepared_disp_fraction[1])

            x_size = tensor.shape[-3] - abs(x_offset)
            y_size = tensor.shape[-2] - abs(y_offset)

            x_start = 0
            y_start = 0
            x_start_misal = 0
            y_start_misal = 0

            if x_offset > 0:
                x_start += x_offset
            else:
                x_start_misal += abs(x_offset)

            if y_offset > 0:
                y_start += y_offset
            else:
                y_start_misal += abs(y_offset)
            # Shift the data of the misaligned portion
            tensor[
                :, x_start : x_start + x_size, y_start : y_start + y_size, z_misal_slice
            ] = tensor[
                :,
                x_start_misal : x_start_misal + x_size,
                y_start_misal : y_start_misal + y_size,
                z_misal_slice,
            ].clone()  # clone necessary bc inplace operation
            # Crop the data from the pad
            result = tensor[:, x_start : x_start + x_size, y_start : y_start + y_size, :]
            return result

        if isinstance(data, dict):
            assert self.keys_to_apply
            for k in self.keys_to_apply:
                assert isinstance(data[k], torch.Tensor)
                data[k] = _process_tensor(data[k])
        else:
            assert isinstance(data, torch.Tensor)
            data = _process_tensor(data)
        return data
