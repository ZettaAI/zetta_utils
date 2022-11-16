from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple, Union

import attrs
import einops
import torch
import torchfields  # pylint: disable=unused-import # monkeypatch

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.layer import Layer
from zetta_utils.layer.volumetric import VolumetricIndex


@builder.register("WarpTaskFactory")
@mazepa.task_factory_cls
@attrs.frozen()
class WarpTaskFactory:
    dst_data_crop: Union[Tuple[int, int, int], List[int]] = (0, 0, 0)
    dst_idx_crop: Optional[Union[Tuple[int, int, int], List[int]]] = None

    # preserve_black: bool = False

    def __call__(
        self,
        idx: VolumetricIndex,
        dst: Layer[Any, VolumetricIndex, torch.Tensor],
        src: Layer[Any, VolumetricIndex, torch.Tensor],
        field: Layer[Any, VolumetricIndex, torch.Tensor],
    ) -> None:

        field_data_raw = field[idx]
        field_data = einops.rearrange(
            field_data_raw,
            "C X Y Z -> Z C X Y"
        ).field() # type: ignore # no type for Torchfields yet

        # TODO: (must) implement translation profiling
        # cc: https://github.com/seung-lab/corgie/blob/main/corgie/stack.py#L191

        src_idx = copy.deepcopy(idx)
        src_data_raw = src[src_idx]
        src_data = einops.rearrange(src_data_raw, "C X Y Z -> Z C X Y")
        # TODO: convert scr_data to float and note the original type
        # convert the result to the original type
        dst_data_raw = field_data.from_pixels()(src_data)
        assert self.dst_data_crop[-1] == 0
        dst_data_cropped = tensor_ops.crop(dst_data_raw, self.dst_data_crop[:-1])  # Z is batch
        dst_data = einops.rearrange(dst_data_cropped, "Z C X Y -> C X Y Z")
        dst_idx = copy.deepcopy(idx)
        if self.dst_idx_crop is not None:
            assert self.dst_idx_crop[-1] == 0
            dst_idx = dst_idx.crop(self.dst_idx_crop)
        else:
            assert self.dst_data_crop[-1] == 0
            dst_idx = dst_idx.crop(self.dst_data_crop)
        dst[dst_idx] = dst_data
