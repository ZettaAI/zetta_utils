# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import floor
from typing import Generic, Optional, Sequence, Tuple, TypeVar, Union, cast

import attrs
import numpy as np

from zetta_utils import builder
from zetta_utils.typing import IntVec3D, Slices3D, Vec3D


def _assert_equal_len(**kwargs: Union[Sequence, Vec3D, IntVec3D]):
    len_map = {k: len(v) for k, v in kwargs.items()}
    if len(set(len_map.values())) != 1:  # means there are unequal lengths
        raise ValueError(
            f"Lengths of all of the '{list(len_map.keys())}' has to be equal. Got: {len_map}"
        )


DEFAULT_UNIT = "nm"

# TODO: Currently both SlicesT and VecT have to be passed independently.
# Ideally, we'd parametrize BoundingBoxND by the number of dimensions,
# and slice and vector types would be infered from it.
# Maybe PEP 646 https://peps.python.org/pep-0646/ can help?

SlicesT = TypeVar("SlicesT", bound=Tuple[slice, ...])
VecT = TypeVar("VecT", bound=Union[Sequence[float], Vec3D])
# @typechecked # https://github.com/agronholm/typeguard/issues/139
@attrs.frozen()
class BoundingBoxND(Generic[SlicesT, VecT]):
    """N-Dimentional cuboid in space.

    Implemented as Generics parametrized by ``SlicesT`` and ``VecT`` types to
    provide exact typing information to static typers.


    """

    bounds: Sequence[Tuple[float, float]]  # Bounding cube bounds, measured in Unit.
    unit: str = DEFAULT_UNIT  # Unit name (for decorative purposes only).

    @property
    def ndim(self) -> int:
        """float of dimensions."""
        return len(self.bounds)

    @classmethod
    def from_slices(
        cls,
        slices: SlicesT,
        resolution: Optional[VecT] = None,
        unit: str = DEFAULT_UNIT,
    ) -> BoundingBoxND[SlicesT, VecT]:
        """Create a :class:`BoundingBoxND` from slices at the given resolution.

        :param slices: Tuple of slices representing a bounding box.
        :param resolution: Resolution at which the slices are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).
        :return: :class:`BoundingBoxND` built according to the specification.

        """
        if resolution is None:
            resolution = cast(VecT, tuple(1 for _ in range(len(slices))))

        _assert_equal_len(
            slices=slices,
            resolution=resolution,
        )

        for s in slices:
            if s.step is not None:
                raise ValueError(f"Cannot construct a boundingbox from strided slice: '{s}'.")

        bounds = tuple(
            (slices[i].start * resolution[i], slices[i].stop * resolution[i])
            for i in range(len(resolution))
        )
        result = cls(bounds=bounds, unit=unit)
        return result

    @classmethod
    def from_coords(
        cls,
        start_coord: VecT,
        end_coord: VecT,
        resolution: VecT,
        unit: str = DEFAULT_UNIT,
    ) -> BoundingBoxND[SlicesT, VecT]:
        """Create a :class:`BoundingBoxND` from start and end coordinates at the given resolution.

        :param start_coord: Tuple representing the start coordinate.
        :param end_coord: Tuple representing the end coordinate.
        :param resolution: Resolution at which the coordinates are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).
        :return: :class:`BoundingBoxND` built according to the specification.

        """
        _assert_equal_len(
            start_coord=start_coord,
            end_coord=end_coord,
            resolution=resolution,
        )
        bounds = tuple(
            (start_coord[i] * resolution[i], end_coord[i] * resolution[i])
            for i in range(len(resolution))
        )
        result = cls(bounds=bounds, unit=unit)
        return result

    def get_slice(
        self,
        dim: int,
        resolution: Union[int, float, VecT],
        allow_slice_rounding: bool = False,
    ) -> slice:
        """Represent the bounding box as a slice along the given dimension.

        :param dim: Dimension along which the slice will be taken.
        :param resolution: Resolution at which the slice will be taken.
        :param allow_slice_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
        :return: Slice representing the bounding box.


        """
        if isinstance(resolution, (int, float)):
            dim_res = resolution
        else:
            dim_res = resolution[dim]

        dim_range_start_raw = self.bounds[dim][0] / dim_res
        dim_range_end_raw = self.bounds[dim][1] / dim_res

        if not allow_slice_rounding:
            if dim_range_start_raw != round(dim_range_start_raw):
                raise ValueError(
                    f"{self} results in slice_start == "
                    f"{dim_range_start_raw} along dimension {dim} "
                    f"at resolution == {resolution} while "
                    "`allow_slice_rounding` == False."
                )
            if dim_range_end_raw != round(dim_range_end_raw):
                raise ValueError(
                    f"{self} results in slice_end == "
                    f"{dim_range_end_raw} along dimension {dim} "
                    f"at resolution == {resolution} while "
                    "`allow_slice_rounding` == False."
                )
        slice_length = int(round(dim_range_end_raw - dim_range_start_raw))
        result = slice(
            floor(dim_range_start_raw),
            floor(dim_range_start_raw) + slice_length,
        )
        return result

    def to_slices(self, resolution: VecT, allow_slice_rounding: bool = False) -> SlicesT:
        """Represent the bounding box as a tuple of slices.

        :param resolution: Resolution at which the slices will be taken.
        :param allow_slice_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
        :return: Slices representing the bounding box.

        """
        result = tuple(
            self.get_slice(i, resolution[i], allow_slice_rounding) for i in range(self.ndim)
        )
        result = cast(SlicesT, result)

        return result

    def crop(
        self,
        crop: Union[Sequence[Union[int, float, tuple[float, float]]], Vec3D, IntVec3D],
        resolution: VecT,
        # in_place: bool = False,
    ) -> BoundingBoxND[SlicesT, VecT]:
        """Create a cropped version of this bounding box.

        :param crop: Specification of how much to crop along each dimension.
        :param resolution: Resolution at which ``crop`` specification was given.
        :return: Cropped bounding box.

        """
        if len(crop) != self.ndim:
            raise ValueError(
                f"Length of the cropping specification ({len(crop)}) != "
                f"BoundingCube ndim ({self.ndim})."
            )

        _assert_equal_len(
            crop=crop,
            bounds=self.bounds,
            resolution=resolution,
        )

        double_sided_crop = []
        for e in crop:
            if isinstance(e, (int, float, np.integer, np.floating)):
                double_sided_crop += [(e, e)]
            else:
                double_sided_crop += [e]

        slices = cast(
            SlicesT,
            tuple(
                slice(
                    self.bounds[i][0] + double_sided_crop[i][0] * resolution[i],
                    self.bounds[i][1] - double_sided_crop[i][1] * resolution[i],
                )
                for i in range(self.ndim)
            ),
        )

        result = BoundingBoxND[SlicesT, VecT].from_slices(
            slices=slices,
            unit=self.unit,
        )

        return result

    def pad(
        self,
        pad: Union[Sequence[Union[float, tuple[float, float]]], Vec3D, IntVec3D],
        resolution: VecT,
        in_place: bool = False,
    ) -> BoundingBoxND[SlicesT, VecT]:
        """Create a padded version of this bounding box.

        :param pad: Specification of how much to pad along each dimension.
        :param resolution: Resolution at which ``pad`` specification was given.
        :param in_place: (WIP) Must be ``False``
        :return: Padded bounding box.

        """
        if len(pad) != self.ndim:
            raise ValueError(
                f"Length of the padding specification ({len(pad)}) != "
                f"BoundingCube ndim ({self.ndim})."
            )

        _assert_equal_len(
            pad=pad,
            bounds=self.bounds,
            resolution=resolution,
        )
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            double_sided_pad = []
            for e in pad:
                if isinstance(e, (int, float, np.integer, np.floating)):
                    double_sided_pad += [(e, e)]
                else:
                    double_sided_pad += [e]
            slices = cast(
                SlicesT,
                tuple(
                    slice(
                        self.bounds[i][0] - double_sided_pad[i][0] * resolution[i],
                        self.bounds[i][1] + double_sided_pad[i][1] * resolution[i],
                    )
                    for i in range(self.ndim)
                ),
            )

            result = BoundingBoxND[SlicesT, VecT].from_slices(
                slices=slices,
                unit=self.unit,
            )

        return result

    def translate(
        self,
        offset: Union[Sequence[float], Vec3D, IntVec3D],
        resolution: Union[Sequence[float], Vec3D, IntVec3D],
        in_place: bool = False,
    ) -> BoundingBoxND[SlicesT, VecT]:
        """Create a translated version of this bounding box.

        :param offset: Specification of how much to translate along each dimension.
        :param resolution: Resolution at which ``offset`` specification was given.
        :param in_place: (WIP) Must be ``False``
        :return: Translated bounding box.

        """
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            slices = cast(
                SlicesT,
                tuple(
                    slice(
                        self.bounds[i][0] + offset[i] * resolution[i],
                        self.bounds[i][1] + offset[i] * resolution[i],
                    )
                    for i in range(self.ndim)
                ),
            )

            result = BoundingBoxND[SlicesT, VecT].from_slices(
                slices=slices,
                unit=self.unit,
            )

        return result

    def pformat(self, resolution: Optional[VecT] = None) -> str:  # pragma: no cover
        """Returns a pretty formatted string for this bounding box at the given
        resolution that is suitable for copying into neuroglancer. For a 3D bcube, the
        string is of the form ``(x_start, y_start, z_start) - (x_end, y_end, z_end)``."""
        if resolution is None:
            if hasattr(self, "resolution"):
                resolution = self.resolution
            else:
                resolution = cast(VecT, tuple(1 for _ in range(self.ndim)))

        slices = self.to_slices(resolution)
        s = ", "
        return (
            f"({s.join([str(slice.start) for slice in slices])})" + " - "
            f"({s.join([str(slice.stop) for slice in slices])})"
        )

    def get_size(self) -> Union[int, float]:  # pragma: no cover
        """Returns the size of the volume in N-D space, in `self.unit^N`."""
        resolution = cast(VecT, tuple(1 for _ in range(self.ndim)))
        slices = self.to_slices(resolution)
        size = 1
        for _, slc in enumerate(slices):
            size *= slc.stop - slc.start
        return size


BoundingCube = BoundingBoxND[Slices3D, Vec3D]  # 3D version of BoundingBoxND
builder.register("BoundingCube", cast_to_vec3d=["start_coord", "end_coord", "resolution"])(
    BoundingCube.from_coords
)


@builder.register("pad_bcube", cast_to_vec3d=["pad_resolution"])
def pad_bcube(
    bcube: BoundingCube,
    pad: Sequence[Union[float, tuple[float, float]]],
    pad_resolution: Vec3D,
) -> BoundingCube:  # pragma: no cover # no logic
    result = bcube.pad(pad=pad, resolution=pad_resolution)
    return result
