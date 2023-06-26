from __future__ import annotations

import math
import random

import attrs
import numpy as np
import torch
from typeguard import typechecked

from zetta_utils import builder
from zetta_utils.distributions import Distribution, to_distribution
from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.tensor_ops import convert

from .common import JointIndexDataAugment
from .transform import DataTransform, GaussianBlur, RandomFill, apply_gaussian_filter


@typechecked
@attrs.mutable
class RandomBox(JointIndexDataAugment):
    key: str
    side: int | Distribution
    density: float | Distribution
    transform: DataTransform

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

    def augment_index(self, idx: VolumetricIndex) -> VolumetricIndex:
        self.prepared_resolution = idx.resolution
        return idx

    def augment_data(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
            raw[tuple(slices)] = self.transform(raw[tuple(slices)])

            # Stop condition
            occluded += intersect.get_size()
            if occluded > goal_in_unit:
                break

        data[self.key] = torch.clamp(raw, min=0, max=1)
        return data

    def prepare(self) -> None:
        self.prepared_shape = self.random_shape()
        self.transform.prepare()


@builder.register("BoxFill")
@typechecked
def build_random_box_fill(
    prob: float,
    key: str,
    side: int | Distribution,
    density: float | Distribution,
    per_box: bool = False,
    is_cube: bool = False,
    fill: float | Distribution = 0,
    fill_per_box: bool = False,
) -> RandomBox:
    return RandomBox(
        prob=prob,
        key=key,
        side=side,
        density=density,
        per_box=per_box,
        is_cube=is_cube,
        transform=RandomFill(fill=fill, frozen=not fill_per_box),
    )


@builder.register("BoxBlur")
@typechecked
def build_random_box_blur(
    prob: float,
    key: str,
    side: int | Distribution,
    density: float | Distribution,
    per_box: bool = False,
    is_cube: bool = False,
    sigma: float | Distribution = 5.0,
    sigma_per_box: bool = False,
) -> RandomBox:
    return RandomBox(
        prob=prob,
        key=key,
        side=side,
        density=density,
        per_box=per_box,
        is_cube=is_cube,
        transform=GaussianBlur(sigma=sigma, frozen=not sigma_per_box),
    )


@typechecked
@attrs.frozen
class _Noise(DataTransform):
    """
    Noise = uniform noise + Gaussian blurring
    """

    sigma0: float = 2.0
    sigma1: float = 5.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
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

    def prepare(self) -> None:
        pass


@builder.register("BoxNoise")
@typechecked
def build_random_box_noise(
    prob: float,
    key: str,
    side: int | Distribution,
    density: float | Distribution,
    per_box: bool = False,
    is_cube: bool = False,
    sigma0: float = 2.0,
    sigma1: float = 5.0,
) -> RandomBox:
    return RandomBox(
        prob=prob,
        key=key,
        side=side,
        density=density,
        per_box=per_box,
        is_cube=is_cube,
        transform=_Noise(sigma0=sigma0, sigma1=sigma1),
    )
