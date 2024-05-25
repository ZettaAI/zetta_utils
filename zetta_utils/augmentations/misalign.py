from __future__ import annotations

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

# This dummy wrappers for mock targets during testing
# trying to decouple implementation from tests a bit


def _select_z(low: int, high: int) -> int:  # pragma: no cover
    return random.randint(low, high)


def _choose_pos_or_neg_misalignent() -> int:  # pragma: no cover
    return random.choice([-1, 1])


def _choose_displacement(low: float, high: float) -> float:
    return random.uniform(low, high)


@builder.register("MisalignProcessor")
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
    mode: Literal["slip", "step"] = "step"
    keys_to_apply: list[str] | None = None

    prepared_disp_fraction: tuple[float, float] | None = attrs.field(init=False, default=None)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        if mode != "read":
            raise NotImplementedError()
        disp_x_in_unit_magn = _choose_displacement(self.disp_min_in_unit, self.disp_max_in_unit)
        disp_y_in_unit_magn = _choose_displacement(self.disp_min_in_unit, self.disp_max_in_unit)

        disp_x_in_unit = disp_x_in_unit_magn * _choose_pos_or_neg_misalignent()
        disp_y_in_unit = disp_y_in_unit_magn * _choose_pos_or_neg_misalignent()

        disp_x_in_idx_res = disp_x_in_unit // idx.resolution[0]
        disp_y_in_idx_res = disp_y_in_unit // idx.resolution[1]

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
            disp_y_in_idx_res / idx.shape[1],
        )
        return idx

    def _get_keys_to_apply_final(self, data: dict[str, Any]) -> list[str]:
        if not self.keys_to_apply:
            if self.mode == "slip":
                raise ValueError(
                    "`keys_to_apply` must be a non-empty list of strings when "
                    "applying `slip` to data of type `dict`"
                )
            # Applied to all tensors by default for step
            assert self.mode == "step"
            result = [k for k, v in data.items() if isinstance(v, torch.Tensor)]
        else:
            result = self.keys_to_apply
        return result

    def _get_chosen_z_considering_crop(
        self, tensor: torch.Tensor, z_size: int, z_chosen: int
    ) -> int | None:
        """
        `None` result means no change of data needs to be done.
        int result means that the data _after_ the indicated Z
        will be misaligned.
        """
        if tensor.size(-1) < z_size:
            z_diff = z_size - tensor.size(-1)
            assert z_diff > 0
            assert z_diff % 2 == 0

            if z_chosen > tensor.size(-1) + z_diff // 2 - 1:
                # the chosen Z fall in the higher cropped out region,
                # so none of the tensor needs to be shifted
                result = None
            elif z_chosen < z_diff // 2 and self.mode == "slip":
                # the chosen Z is falls in the lower cropped out region,
                # so do nothing for slip
                result = None
            else:
                result = z_chosen - z_diff // 2
                if result < 0:
                    assert self.mode == "step"
                    # "step" when the chosen Z falls in the lower cropped
                    # out region, shift the whole tensor
                    result = 0
        else:
            result = z_chosen

        return result

    def _process_tensor(self, tensor: torch.Tensor, z_size: int, z_chosen: int) -> torch.Tensor:
        this_z_chosen = self._get_chosen_z_considering_crop(
            tensor, z_size=z_size, z_chosen=z_chosen
        )

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

        if this_z_chosen is not None:
            if self.mode == "slip":
                z_misal_slice = slice(this_z_chosen, this_z_chosen + 1)
            else:
                z_misal_slice = slice(this_z_chosen, tensor.size(-1))

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

    def process_data(self, data: T, mode: Literal["read", "write"]) -> T:
        if mode != "read":
            raise NotImplementedError()

        keys_to_apply_final: list[str] | None = None
        if isinstance(data, torch.Tensor):
            z_size = data.shape[-1]
        else:
            keys_to_apply_final = self._get_keys_to_apply_final(data)
            sizes = [data[k].shape for k in keys_to_apply_final]
            z_size = max(size[-1] for size in sizes)

        z_chosen = _select_z(0, z_size - 1)

        if isinstance(data, dict):
            assert keys_to_apply_final
            for k in keys_to_apply_final:
                assert isinstance(data[k], torch.Tensor)
                data[k] = self._process_tensor(data[k], z_size=z_size, z_chosen=z_chosen)
        else:
            data = self._process_tensor(data, z_size=z_size, z_chosen=z_chosen)
        return data
