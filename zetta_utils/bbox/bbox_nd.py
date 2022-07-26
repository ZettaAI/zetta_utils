# pylint: disable=missing-docstring, no-else-raise
from __future__ import annotations

from math import floor
from typing import Union, Sequence, Tuple, Optional

import attrs
from typeguard import typechecked
from zetta_utils.typing import Number
from zetta_utils.bbox import DEFAULT_UNIT


@typechecked  # typechecking disabled because of subscripted generics
@attrs.frozen()
class BoundingBoxND:
    """Represents a N-D cuboid in space."""

    bounds: Sequence[Tuple[Number, Number]]
    unit: str = DEFAULT_UNIT
    expected_ndim: Optional[int] = None

    @classmethod
    def validate_lengths(cls, **kwargs):
        len_map = {k: len(v) for k, v in kwargs.items()}

        if cls.expected_ndim is None:
            if not all(e == cls.expected_ndim for e in len_map.values()):
                raise ValueError(
                    f"Lengths of all of the '{list(len_map.keys())}' has to be "
                    "equal to {self.expected_ndim}. Got: {len_map}"
                )
        else:
            if len(set(len_map.values())) != 1:  # there are unequal lengths
                raise ValueError(
                    f"Lengths of all of the '{list(len_map.keys())}' has to be "
                    "equal. Got: {len_map}"
                )

    @property
    def ndim(self) -> int:
        return len(self.bounds)

    @classmethod
    def from_slices(
        cls,
        slices: Sequence[slice],
        resolution: Sequence[Number],
        unit: str = DEFAULT_UNIT,
    ) -> BoundingBoxND:
        cls.validate_lengths(
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
        result = cls(bounds=bounds, unit=unit, expected_ndim=len(bounds))
        return result

    @classmethod
    def from_coords(
        cls,
        start_coord: Sequence[Number],
        end_coord: Sequence[Number],
        resolution: Sequence[Number],
        unit: str = DEFAULT_UNIT,
    ) -> BoundingBoxND:
        cls.validate_lengths(
            start_coord=start_coord,
            end_coord=end_coord,
            resoluiton=resolution,
        )
        bounds = tuple(
            (start_coord[i] * resolution[i], end_coord[i] * resolution[i])
            for i in range(len(resolution))
        )
        result = cls(bounds=bounds, unit=unit, expected_ndim=len(bounds))
        return result

    def get_slice(
        self,
        dim: int,
        resolution: Union[int, float, Sequence[Number]],
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

    def to_slices(
        self, resolution: Sequence[Number], allow_rounding: bool = False
    ) -> Sequence[slice]:
        result = tuple(self.get_slice(i, resolution[i], allow_rounding) for i in range(self.ndim))

        return result

    def pad(
        self,
        pad: Sequence[Union[int, float, tuple[Union[int, float], Union[int, float]]]],
        resolution: Sequence[Number],
        in_place: bool = False,
    ) -> BoundingBoxND:
        if len(pad) != self.ndim:
            raise ValueError(
                f"Length of the padding specification ({len(pad)}) != "
                f"BoundingCube ndim ({self.ndim})."
            )

        self.validate_lengths(
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

            result = BoundingBoxND.from_slices(
                slices=[
                    slice(
                        self.bounds[i][0] - double_sided_pad[i][0] * resolution[i],
                        self.bounds[i][1] + double_sided_pad[i][1] * resolution[i],
                    )
                    for i in range(self.ndim)
                ],
                resolution=tuple(1 for _ in range(self.ndim)),
                unit=self.unit,
            )
        return result

    def translate(
        self,
        offset: Sequence[Number],
        resolution: Sequence[Number],
        in_place: bool = False,
    ) -> BoundingBoxND:
        if in_place:
            raise NotImplementedError  # pragma: no cover
        else:
            result = BoundingBoxND.from_slices(
                slices=tuple(
                    slice(
                        self.bounds[i][0] + offset[i] * resolution[i],
                        self.bounds[i][1] + offset[i] * resolution[i],
                    )
                    for i in range(self.ndim)
                ),
                resolution=tuple(1 for _ in range(self.ndim)),
                unit=self.unit,
            )

        return result
