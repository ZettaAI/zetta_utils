# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import pytest
import torch

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricIndexStartOffsetOverrider,
)
from zetta_utils.layer.volumetric.tools import ROIMaskProcessor


@pytest.mark.parametrize(
    "resolution, stride, offset, idx, stride_start_offset_in_unit, mode, chunk1",
    [
        [
            Vec3D(1, 1, 1),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 10), (0, 10)))),
            None,
            "expand",
            BBox3D(((2, 4), (0, 3), (0, 5))),
        ],
        [
            Vec3D(1, 1, 1),
            None,
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 10), (0, 10)))),
            None,
            "expand",
            BBox3D(((2, 4), (0, 3), (0, 5))),
        ],
        [
            Vec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 10), (0, 10)))),
            None,
            "expand",
            BBox3D(((4, 8), (0, 6), (0, 10))),
        ],
        [
            Vec3D(3, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 15), (0, 15)))),
            None,
            "expand",
            BBox3D(((6, 12), (8, 14), (4, 14))),
        ],
        [
            Vec3D(3, 2, 2),
            IntVec3D(2, 2, 2),
            IntVec3D(2, 2, 2),
            VolumetricIndex(Vec3D(1, 1, 1), BBox3D(((0, 10), (0, 15), (0, 15)))),
            IntVec3D(1, 2, 3),
            "expand",
            BBox3D(((7, 13), (2, 8), (3, 13))),
        ],
        [
            None,
            IntVec3D(2, 2, 2),
            IntVec3D(0, 0, 0),
            VolumetricIndex(Vec3D(3, 2, 1), BBox3D(((0, 16), (0, 16), (0, 16)))),
            None,
            "expand",
            BBox3D(((6, 12), (0, 6), (0, 5))),
        ],
    ],
)
def test_volumetric_index_chunker(
    resolution, stride, offset, idx, stride_start_offset_in_unit, mode, chunk1
):
    vic = VolumetricIndexChunker(
        chunk_size=IntVec3D(2, 3, 5),
        max_superchunk_size=None,
        stride=stride,
        resolution=resolution,
        offset=offset,
    )
    res = list(vic(idx=idx, stride_start_offset_in_unit=stride_start_offset_in_unit, mode=mode))
    assert res[1].bbox == chunk1


@pytest.mark.parametrize(
    "override_offset, expected_start, expected_stop",
    [
        [[7, 8, 9], [7, 8, 9], [17, 18, 19]],
        [[None, None, 10], [4, 5, 10], [14, 15, 20]],
        [[None, None, None], [4, 5, 6], [14, 15, 16]],
    ],
)
def test_volumetric_index_offset_overrider(override_offset, expected_start, expected_stop):
    index = VolumetricIndex(resolution=Vec3D(1, 1, 1), bbox=BBox3D(((4, 14), (5, 15), (6, 16))))
    visoo = VolumetricIndexStartOffsetOverrider(override_offset=override_offset)
    index = visoo(index)
    assert index.start == Vec3D(*expected_start)
    assert index.stop == Vec3D(*expected_stop)


@pytest.mark.parametrize(
    "start_coord, end_coord, resolution, targets, data_shape, expected_mask_region, existing_masks",
    [
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            [],
        ),
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            ["target1"],
        ),
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2", "target3"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            ["target1", "target3"],
        ),
    ],
)
def test_roi_mask_processor_read(
    start_coord, end_coord, resolution, targets, data_shape, expected_mask_region, existing_masks
):
    processor = ROIMaskProcessor(
        start_coord=start_coord,
        end_coord=end_coord,
        resolution=resolution,
        targets=targets,
    )

    idx = VolumetricIndex.from_coords(
        start_coord=Vec3D(*start_coord), end_coord=Vec3D(*end_coord), resolution=Vec3D(*resolution)
    )
    processor.process_index(idx, "read")

    data = {target: torch.rand(*data_shape) for target in targets}
    for target in existing_masks:
        data[target + "_mask"] = torch.ones(*data_shape)  # Pre-existing mask for the target

    processed_data = processor.process_data(data, "read")

    for target in targets:
        assert target in processed_data

        if target in existing_masks:
            assert torch.all(processed_data[target + "_mask"] == data[target + "_mask"])
        else:
            mask = processed_data[target + "_mask"]
            inside_roi = mask[
                0, expected_mask_region[0], expected_mask_region[1], expected_mask_region[2]
            ]
            assert torch.all(inside_roi == 1)

            full_slice = slice(None)
            outside_roi_slices = tuple(
                full_slice if s == full_slice else slice(None, s.start)
                for s in expected_mask_region
            )
            outside_roi = mask[
                0, outside_roi_slices[0], outside_roi_slices[1], outside_roi_slices[2]
            ]
            assert torch.all(outside_roi == 0)
