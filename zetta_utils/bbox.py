# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import floor
from typing import Union, Sequence, Tuple, Generic, TypeVar, Optional, cast

import attrs

from zetta_utils import spec_parser
from zetta_utils.typing import Number


def assert_equal_len(**kwargs: Sequence):
    len_map = {k: len(v) for k, v in kwargs.items()}
    if len(set(len_map.values())) != 1:  # means there are unequal lengths
        raise ValueError(
            f"Lengths of all of the '{list(len_map.keys())}' has to be equal. Got: {len_map}"
        )


DEFAULT_UNIT = "nm"

# TODO: Currently both SlicesT and VecT have to be passed independently.
# Ideally, we'd parametrize BoundingBoxND by the number of dimensions,
# and slice and vector types would be infered from it.
# (PEP 646 https://peps.python.org/pep-0646/ )

SlicesT = TypeVar("SlicesT", bound=Tuple[slice, ...])
VecT = TypeVar("VecT", bound=Tuple[Number, ...])
# @typechecked # https://github.com/agronholm/typeguard/issues/139
@attrs.frozen()
class BoundingBoxND(Generic[SlicesT, VecT]):
    """Represents a N-D cuboid in space."""

    bounds: Sequence[Tuple[Number, Number]]
    unit_name: str = DEFAULT_UNIT

    @property
    def ndim(self) -> int:
        return len(self.bounds)

    @classmethod
    def from_slices(
        cls,
        slices: SlicesT,
        resolution: Optional[VecT] = None,
        unit_name: str = DEFAULT_UNIT,
    ) -> BoundingBoxND[SlicesT, VecT]:
        if resolution is None:
            resolution = cast(VecT, tuple(1 for _ in range(len(slices))))

        assert_equal_len(
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
        result = cls(bounds=bounds, unit_name=unit_name)
        return result

    @classmethod
    def from_coords(
        cls,
        start_coord: VecT,
        end_coord: VecT,
        resolution: VecT,
        unit_name: str = DEFAULT_UNIT,
    ) -> BoundingBoxND[SlicesT, VecT]:
        assert_equal_len(
            start_coord=start_coord,
            end_coord=end_coord,
            resoluiton=resolution,
        )
        bounds = tuple(
            (start_coord[i] * resolution[i], end_coord[i] * resolution[i])
            for i in range(len(resolution))
        )
        result = cls(bounds=bounds, unit_name=unit_name)
        return result

    def get_slice(
        self,
        dim: int,
        resolution: Union[int, float, VecT],
        allow_rounding: bool = False,
    ) -> slice:
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
        result = tuple(self.get_slice(i, resolution[i], allow_rounding) for i in range(self.ndim))
        result = cast(SlicesT, result)

        return result

    def pad(
        self,
        pad: Sequence[Union[int, float, tuple[Union[int, float], Union[int, float]]]],
        resolution: VecT,
        in_place: bool = False,
    ) -> BoundingBoxND[SlicesT, VecT]:
        if len(pad) != self.ndim:
            raise ValueError(
                f"Length of the padding specification ({len(pad)}) != "
                f"BoundingCube ndim ({self.ndim})."
            )

        assert_equal_len(
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
                unit_name=self.unit_name,
            )

        return result

    def translate(
        self,
        offset: Sequence[Union[int, float]],
        resolution: Sequence[Union[int, float]],
        in_place: bool = False,
    ) -> BoundingBoxND[SlicesT, VecT]:
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
                unit_name=self.unit_name,
            )

        return result

    def __repr__(self) -> str:  # pragma: no cover
        parts = ["BoundingCube("]
        for i in range(self.ndim):  # pylint: disable=unused-variable
            parts.append(f"[Dim {i}] {self.bounds[i][0]}:{self.bounds[i][1]}{self.unit_name},")
        parts.append(")")
        result = "".join(parts)
        return result


BoundingBox = BoundingBoxND[Tuple[slice, slice], Tuple[Number, Number]]

BoundingCube = BoundingBoxND[Tuple[slice, slice, slice], Tuple[Number, Number, Number]]
spec_parser.register("BoundingCube")(BoundingCube.from_coords)
