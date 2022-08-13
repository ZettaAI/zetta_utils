# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import floor
from typing import Union, Sequence, Tuple, Generic, TypeVar, Optional, cast

import attrs

from zetta_utils import builder
from zetta_utils.common.typing import Number, Vec3D, Slices3D


def _assert_equal_len(**kwargs: Sequence):
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
VecT = TypeVar("VecT", bound=Sequence[Number])
# @typechecked # https://github.com/agronholm/typeguard/issues/139
@attrs.frozen()
class BoundingBoxND(Generic[SlicesT, VecT]):
    """N-Dimentional cuboid in space.

    Implemented as Generics parametrized by ``SlicesT`` and ``VecT`` types to
    provide exact typing information to static typers.


    """

    bounds: Sequence[Tuple[Number, Number]]  # Bounding cube bounds, measured in Unit.
    unit: str = DEFAULT_UNIT  # Unit name (for decorative purposes only).

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.bounds)

    @classmethod
    def from_slices(
        cls,
        slices: SlicesT,
        resolution: Optional[VecT] = None,
        unit: str = DEFAULT_UNIT,
    ) -> BoundingBoxND[SlicesT, VecT]:
        """Create a :class:`BoundingBoxND` from slices at the given resolution.

        :param slices: Tuple of slices represeinting a bounding box.
        :param resolution: Resolution at which the slices are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).
        :return: :class:`BoundingBoxND` built according to the specification.

        """
        if resolution is None:
            resolution = cast(VecT, tuple(1 for _ in range(len(slices))))

        _assert_equal_len(
            slices=slices,
            resoluiton=resolution,
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

        :param start_coord: Tuple represeting the start coordinate.
        :param end_coord: Tuple represeting the end coordinate.
        :param resolution: Resolution at which the coordinates are given.
            If not given, assumed to be unit resolution.
        :param unit: Unit name (decorative purposes only).
        :return: :class:`BoundingBoxND` built according to the specification.

        """
        _assert_equal_len(
            start_coord=start_coord,
            end_coord=end_coord,
            resoluiton=resolution,
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
        allow_rounding: bool = False,
    ) -> slice:
        """Represent the bounding box as a slice along the given dimension.

        :param dim: Dimension along which the slice will be taken.
        :param resolution: Resolution at which the slice will be taken.
        :param allow_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
        :return: Slice representing the bounding box.


        """
        if isinstance(resolution, (int, float)):
            dim_res = resolution
        else:
            dim_res = resolution[dim]

        dim_range_start_raw = self.bounds[dim][0] / dim_res
        dim_range_end_raw = self.bounds[dim][1] / dim_res

        if not allow_rounding:
            if dim_range_start_raw != round(dim_range_start_raw):
                raise ValueError(
                    f"{self} results in slice_start == "
                    f"{dim_range_start_raw} along dimension {dim} "
                    f"at resolution == {resolution} while "
                    "`allow_rounding` == False."
                )
            if dim_range_end_raw != round(dim_range_end_raw):
                raise ValueError(
                    f"{self} results in slice_end == "
                    f"{dim_range_end_raw} along dimension {dim} "
                    f"at resolution == {resolution} while "
                    "`allow_rounding` == False."
                )
        slice_length = int(round(dim_range_end_raw - dim_range_start_raw))
        result = slice(
            floor(dim_range_start_raw),
            floor(dim_range_start_raw) + slice_length,
        )
        return result

    def to_slices(self, resolution: VecT, allow_rounding: bool = False) -> SlicesT:
        """Represent the bounding box as a tuple of slices.

        :param resolution: Resolution at which the slices will be taken.
        :param allow_rounding: Whether to allow representing bounding box
            with non-integer slice start/end at the given resolution.
        :return: Slices representing the bounding box.

        """
        result = tuple(self.get_slice(i, resolution[i], allow_rounding) for i in range(self.ndim))
        result = cast(SlicesT, result)

        return result

    def pad(
        self,
        pad: Sequence[Union[int, float, tuple[Union[int, float], Union[int, float]]]],
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
            resoluiton=resolution,
        )
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            double_sided_pad = []
            for i in pad:
                if isinstance(i, (int, float)):
                    double_sided_pad += [(i, i)]
                else:
                    double_sided_pad += [i]
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
        offset: Sequence[Union[int, float]],
        resolution: Sequence[Union[int, float]],
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


BoundingCube = BoundingBoxND[Slices3D, Vec3D]  # 3D version of BoundingBoxND
builder.register("BoundingCube")(BoundingCube.from_coords)
