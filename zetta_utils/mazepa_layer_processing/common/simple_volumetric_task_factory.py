from __future__ import annotations

import copy
from typing import Any, Callable, Generic, List, Optional, Tuple, Union

import attrs
import torch
from typing_extensions import ParamSpec

from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.layer import Layer
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.typing import Vec3D

P = ParamSpec("P")


@builder.register("SimpleVolumetricTaskFactory")
@mazepa.task_factory_cls
@attrs.frozen(init=False)
class SimpleVolumetricTaskFactory(Generic[P]):
    """
    Wrapper that converts a volumetric processing callable to a task factory.
    Adds support for data cropping, index cropping and index resolution change
    functionalities.

    :param fn: Callable that will perform data processing
    :param dst_data_crop: Output crop along XYZ dimensions.
    :param dst_idx_res: What resolution to write output at. If ``None``,
        will write at the input resolution.
    """

    fn: Callable[P, torch.Tensor]

    dst_data_crop: Union[Tuple[int, int, int], List[int]] = (0, 0, 0)
    dst_idx_res: Optional[Vec3D] = None
    dst_idx_crop: Optional[Union[Tuple[int, int, int], List[int]]] = None
    # download_layers: bool = True # Could be made optoinal

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        # TODO: Is it possible to make mypy check this statically?
        # try task_factory_with_idx_cls
        assert len(args) == 0
        assert "idx" in kwargs
        assert "dst" in kwargs

        idx: VolumetricIndex = kwargs["idx"]  # type: ignore
        dst: Layer[Any, VolumetricIndex] = kwargs["dst"]  # type: ignore

        # TODO: assert that types check out
        fn_kwargs = {}
        for k, v in kwargs.items():
            if k in ["dst", "idx"]:
                pass
            elif isinstance(v, Layer):
                fn_kwargs[f"{k}_data"] = v[idx]
            else:
                fn_kwargs[k] = v

        result_raw = self.fn(**fn_kwargs)
        dst_data = tensor_ops.crop(result_raw, crop=self.dst_data_crop)

        dst_idx = copy.deepcopy(idx)

        if self.dst_idx_res is not None:
            dst_idx.resolution = self.dst_idx_res

        if self.dst_idx_crop is not None:
            dst_idx = dst_idx.crop(self.dst_idx_crop)
        else:
            dst_idx = dst_idx.crop(self.dst_data_crop)
        dst[dst_idx] = dst_data
