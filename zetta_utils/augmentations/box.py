from __future__ import annotations

import math
import random
from abc import abstractmethod
from typing import Literal

import attrs
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer import JointIndexDataProcessor
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.tensor_ops import convert

from .blur import apply_gaussian_filter


@typechecked
@attrs.mutable
class RandomBoxAugment(JointIndexDataProcessor):
    key: str
    side: int | Distribution
    density: float | Distribution

    per_box: bool = False
    is_cube: bool = False

    side_distr: Distribution = attrs.field(init=False)
    density_distr: Distribution = attrs.field(init=False)

    prepared_shape: Vec3D | None = attrs.field(init=False, default=None)
    prepared_resolution: Vec3D | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        self.side_distr = to_distribution(self.side)
        self.density_distr = to_distribution(self.density)

    def random_shape(self) -> Vec3D:
        if self.is_cube:
            result = Vec3D(*([int(self.side_distr())] * 3))
        else:
            result = Vec3D(*[int(self.side_distr()) for _ in range(3)])
        return result

    @staticmethod
    def random_location(roi: BBox3D) -> Vec3D:
        return Vec3D(*[random.randint(0, dim - 1) for dim in roi.shape])

    def process_index(
        self, idx: VolumetricIndex, mode: Literal["read", "write"]
    ) -> VolumetricIndex:
        self.prepared_resolution = idx.resolution
        return idx

    def process_data(
        self, data: dict[str, torch.Tensor], mode: Literal["read", "write"]
    ) -> dict[str, torch.Tensor]:
        assert self.key in data
        raw = data[self.key]

        # Prepare for processing
        self.prepare()

        # Random density
        density: float = self.density_distr()
        assert 0 <= density <= 1

        # Index resolution
        assert self.prepared_resolution is not None
        resolution: Vec3D = self.prepared_resolution

        # BBox for raw data
        roi = BBox3D.from_coords(
            start_coord=(0, 0, 0),
            end_coord=tuple(dim for dim in raw.shape[-3:]),
            resolution=tuple(resolution),
        )

        # Stop condition
        occluded: float = 0
        goal_in_unit: float = roi.get_size() * density
        num_voxels: int = math.prod(raw.shape[-3:])
        max_iterations = int(num_voxels * density)

        # Generate and process random boxes
        for _ in range(max_iterations):

            # Random box shape
            if self.per_box:
                shape: Vec3D = self.random_shape()
            else:
                assert self.prepared_shape is not None
                shape = self.prepared_shape

            # Random location
            loc: Vec3D = self.random_location(roi)

            # Generate a box to occlude, centered on `loc`
            assert shape is not None
            box_in_unit = BBox3D.from_coords(
                start_coord=tuple(loc),
                end_coord=tuple(loc + shape),
                resolution=(1, 1, 1),
            ).translated(
                offset=tuple(-(shape // 2)),  # pylint: disable=invalid-unary-operand-type
                resolution=(1, 1, 1),
            )

            # Intersection
            intersect = roi.intersection(box_in_unit)
            idx = intersect.to_slices(
                resolution=tuple(resolution),
                allow_slice_rounding=True,
            )
            extra_dims = raw.ndim - len(idx)
            slices = [slice(0, None) for _ in range(extra_dims)]
            slices += list(idx)

            # Process the box
            raw[tuple(slices)] = self.process_box(raw[tuple(slices)])

            # Stop condition
            occluded += intersect.get_size()
            if occluded > goal_in_unit:
                break

        data[self.key] = torch.clamp(raw, min=0, max=1)
        return data

    def prepare(self) -> None:
        self.prepared_shape = self.random_shape()

    @abstractmethod
    def process_box(self, data: torch.Tensor) -> torch.Tensor:
        ...


@builder.register("FillBoxAugment")
@typechecked
@attrs.mutable
class FillBoxAugment(RandomBoxAugment):
    fill: float | Distribution = 0
    fill_per_box: bool = False

    fill_distr: Distribution = attrs.field(init=False)

    prepared_fill: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.fill_distr = to_distribution(self.fill)

    def prepare(self) -> None:
        super().prepare()
        self.prepared_fill = self.fill_distr()

    def process_box(self, data: torch.Tensor) -> torch.Tensor:
        if self.fill_per_box:
            value = self.fill_distr()  # type: float
        else:
            assert self.prepared_fill is not None
            value = self.prepared_fill
        result = data.fill_(value)
        return result


@builder.register("BlurBoxAugment")
@typechecked
@attrs.mutable
class BlurBoxAugment(RandomBoxAugment):
    sigma: float | Distribution = 5.0
    sigma_per_box: bool = False

    sigma_distr: Distribution = attrs.field(init=False)

    prepared_sigma: float | None = attrs.field(init=False, default=None)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.sigma_distr = to_distribution(self.sigma)

    def prepare(self) -> None:
        super().prepare()
        self.prepared_sigma = self.sigma_distr()

    def process_box(self, data: torch.Tensor) -> torch.Tensor:
        if self.sigma_per_box:
            sigma = self.sigma_distr()  # type: float
        else:
            assert self.prepared_sigma is not None
            sigma = self.prepared_sigma
        result = apply_gaussian_filter(data, sigma)
        return result


@builder.register("NoiseBoxAugment")
@typechecked
@attrs.mutable
class NoiseBoxAugment(RandomBoxAugment):
    """
    Noise = uniform noise + Gaussian blurring
    """

    sigma0: float = 2.0
    sigma1: float = 5.0

    def process_box(self, data: torch.Tensor) -> torch.Tensor:
        dtype = convert.to_np(data).dtype
        shape = data.shape[-3:]

        # Stage 1
        patch = np.random.rand(*shape).astype(dtype)
        sigma = (self.sigma0, self.sigma0, 0)
        patch = apply_gaussian_filter(patch, sigma)

        # Stage 2
        patch = (patch > 0.5).astype(dtype)
        sigma = (self.sigma1, self.sigma1, 0)
        patch = apply_gaussian_filter(patch, sigma)
        data[..., :, :, :] = convert.astype(patch, data, cast=True)

        return data
