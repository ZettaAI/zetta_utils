# pylint: disable=missing-docstring
from typing import Tuple, Literal, Optional, Union, Iterable, Callable
import attrs
from typeguard import typechecked

import zetta_utils as zu
from zetta_utils.data.layers.indexers.base import BaseIndexer
from zetta_utils.typing import Vec3D, Slices3D
from zetta_utils.data.basic_ops import InterpolationMode
from zetta_utils.data.processors import Interpolate


VolumetricDimFormat = Literal["bcxyz"]


RawVolumetricIndex = Union[
    Slices3D,
    Tuple[Optional[Vec3D], slice, slice, slice],
]


@typechecked
@attrs.frozen
class VolumetricIndex:  # pylint: disable=too-few-public-methods
    resolution: Vec3D
    slices: Slices3D



@typechecked
@attrs.mutable
class VolumetricIndexer(BaseIndexer): # pylint: disable=too-few-public-methods
    index_resolution: Optional[Vec3D] = None
    data_resolution: Optional[Vec3D] = None
    interpolation_mode: Optional[InterpolationMode] = None
    dims: VolumetricDimFormat = "bcxyz"

    def _process_idx(self, idx: RawVolumetricIndex) -> Tuple[VolumetricIndex, Vec3D]:
        if len(idx) == 3:  # Tuple[slice, slice, sclie], default index
            specified_resolution = None
            slices_raw = idx  # type: Tuple[slice, slice, slice] # type: ignore
        else:
            assert len(idx) == 4
            specified_resolution = idx[0]  # type: Vec3D # type: ignore
            slices_raw = idx[1:]  # type: ignore # Dosn't know the idx[1:] type

        if self.index_resolution is not None:
            index_resolution = self.index_resolution
        elif specified_resolution is not None:
            index_resolution = specified_resolution
        else:
            raise ValueError(
                "Neither IO operation resolution nor default `index_resolution` is provided"
            )
        bcube = zu.bcube.BoundingCube.from_slices(slices=slices_raw, resolution=index_resolution)

        if specified_resolution is not None:
            desired_resolution = specified_resolution
        elif self.data_resolution is not None:
            desired_resolution = self.data_resolution
        else:
            assert self.index_resolution is not None
            desired_resolution = self.index_resolution

        if self.data_resolution is not None:
            data_resolution = self.data_resolution
        else:
            data_resolution = desired_resolution

        slices_final = bcube.to_slices(data_resolution)  # type: Slices3D # type: ignore

        idx_final = VolumetricIndex(
            resolution=data_resolution,
            slices=slices_final,
        )

        return idx_final, desired_resolution

    def __call__(
        self, idx: RawVolumetricIndex, mode: Literal["read", "write"]
    ) -> Tuple[VolumetricIndex, Iterable[Callable]]:
        idx_final, desired_resolution = self._process_idx(idx)
        processors = []
        if idx_final.resolution != desired_resolution:
            if self.interpolation_mode is None:
                raise RuntimeError(
                    "Don't know how to interpolate images: `data_resolution` differs from "
                    "the desired IO resolution while `interpolation_mode` == None"
                )
            if self.dims != "bcxyz":  # pragma: no cover
                raise NotImplementedError
            if mode == "read":
                scale_factor = tuple(
                    idx_final.resolution[i] / desired_resolution[i] for i in range(3)
                )
            else:
                scale_factor = tuple(
                    desired_resolution[i] / idx_final.resolution[i] for i in range(3)
                )

            processors.append(
                Interpolate(scale_factor=scale_factor, interpolation_mode=self.interpolation_mode)
            )

        return idx_final, processors
