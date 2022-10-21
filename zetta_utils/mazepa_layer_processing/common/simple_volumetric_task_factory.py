from __future__ import annotations
from typing import Any, Generic, Callable, Optional
import copy
from typing_extensions import ParamSpec
import torch
import attrs
from zetta_utils import builder, mazepa, tensor_ops
from zetta_utils.layer import Layer
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.typing import Vec3D

P = ParamSpec("P")


@builder.register("SimpleVolumetricTaskFactory")
@mazepa.task_factory_cls
@attrs.frozen()
class SimpleVolumetricTaskFactory(Generic[P]):
    """
    Provides data cropping, index cropping and index resolution change
    functionalities for volumetric data processing.
    :param fn: Callable that will perform data processing
    :param dst_data_crop: Crop along XYZ dimensions of the callable output.
    :param dst_idx_res: What resolution to write at. If ``None``, will
        default to ``dst_data_crop``.
    """

    fn: Callable[P, torch.Tensor]

    dst_data_crop: tuple[int, ...] | list[int] = (0, 0, 0)
    dst_idx_res: Optional[Vec3D] = None
    dst_idx_crop: Optional[tuple[int, ...] | list[int]] = None
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
