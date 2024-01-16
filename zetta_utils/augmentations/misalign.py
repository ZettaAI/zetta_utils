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


@builder.register("SlipMisalign")
@typechecked
@attrs.mutable
class MisalignProcessor(JointIndexDataProcessor[T, VolumetricIndex]):
    prob: float
    disp_min: int
    disp_max: int
    mode: Literal["slip", "step"] = "slip"
    keys_to_apply: list[str] | None = None

    prepared_disp: tuple[int, int] | None = attrs.field(init=False, default=None)

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        if mode != "read":
            raise NotImplementedError()
        disp_x = random.randint(self.disp_min, self.disp_max) * random.choice([1, -1])
        disp_y = random.randint(self.disp_min, self.disp_max) * random.choice([1, -1])
        self.prepared_disp = (disp_x, disp_y)

        start_offset = [0, 0, 0]
        end_offset = [0, 0, 0]
        if disp_x > 0:
            start_offset[0] = -disp_x
        else:
            end_offset[0] = -disp_x
        if disp_y > 0:
            start_offset[1] = -disp_y
        else:
            end_offset[1] = -disp_y

        idx = idx.translated_start(Vec3D[int](*start_offset)).translated_end(
            Vec3D[int](*end_offset)
        )
        return idx

    def _get_tensor_shape(self, data: T) -> torch.Size:
        if isinstance(data, torch.Tensor):
            result = data.shape
        else:
            assert isinstance(data, dict)
            if not self.keys_to_apply:
                raise ValueError(
                    "`keys_to_apply` must be a non-empty list of springs when "
                    "applying to data of type `dict`"
                )
            tensor_shapes = [data[k].shape for k in self.keys_to_apply]
            if not all(e == tensor_shapes[0] for e in tensor_shapes):
                raise ValueError(
                    "Tensor shapes to be processed with misalignment augmentation "
                    f"must all have the same shape. Got keys: {self.keys_to_apply} "
                    f"shapes: {tensor_shapes}"
                )
            result = tensor_shapes[0]
        return result

    def process_data(self, data: T, mode: Literal["read", "write"]) -> T:
        assert self.prepared_disp is not None
        if mode != "read":
            raise NotImplementedError()

        tensor_shape = self._get_tensor_shape(data)

        z_chosen = random.randint(0, tensor_shape[-1] - 1)
        if self.mode == "slip":
            z_misal_slice = slice(z_chosen, z_chosen + 1)
        else:
            z_misal_slice = slice(z_chosen, tensor_shape[-1])

        x_size = tensor_shape[1] - abs(self.prepared_disp[0])
        y_size = tensor_shape[2] - abs(self.prepared_disp[1])

        x_start = 0
        y_start = 0
        x_start_misal = 0
        y_start_misal = 0

        if self.prepared_disp[0] > 0:
            x_start += self.prepared_disp[0]
        else:
            x_start_misal += abs(self.prepared_disp[0])

        if self.prepared_disp[1] > 0:
            y_start += self.prepared_disp[1]
        else:
            y_start_misal += abs(self.prepared_disp[1])

        def _process_tensor(tensor: torch.Tensor) -> torch.Tensor:
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
