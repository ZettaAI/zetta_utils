# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import numpy as np
import pytest

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import (
    VolumetricIndex,
    VolumetricIndexChunker,
    VolumetricIndexOverrider,
    VolumetricIndexScaler,
)
from zetta_utils.layer.volumetric.tools import ROIMaskProcessor


@pytest.mark.parametrize(
    "resolution, stride, offset, idx, stride_start_offset, mode, chunk1",
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
            IntVec3D(3, 4, 6),
            "expand",
            BBox3D(((9, 15), (4, 10), (4, 14))),
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
    resolution, stride, offset, idx, stride_start_offset, mode, chunk1
):
    vic = VolumetricIndexChunker(
        chunk_size=IntVec3D(2, 3, 5),
        max_superchunk_size=None,
        stride=stride,
        resolution=resolution,
        offset=offset,
    )
    res = list(vic(idx=idx, stride_start_offset=stride_start_offset, mode=mode))
    assert res[1].bbox == chunk1


@pytest.mark.parametrize(
    """override_offset, override_size, override_resolution,
    expected_start, expected_stop, expected_resolution""",
    [
        [[7, 8, 9], None, None, [7, 8, 9], [17, 18, 19], [1, 1, 1]],
        [[None, None, 10], None, None, [4, 5, 10], [14, 15, 20], [1, 1, 1]],
        [[None, None, None], None, None, [4, 5, 6], [14, 15, 16], [1, 1, 1]],
        [None, [7, 8, 9], None, [4, 5, 6], [11, 13, 15], [1, 1, 1]],
        [None, [None, None, 8], None, [4, 5, 6], [14, 15, 14], [1, 1, 1]],
        [None, [None, None, None], None, [4, 5, 6], [14, 15, 16], [1, 1, 1]],
        [None, None, [2, 2, 2], [4, 5, 6], [14, 15, 16], [2, 2, 2]],
        [None, None, [None, None, 2.1], [4, 5, 6], [14, 15, 16], [1, 1, 2.1]],
        [None, None, [None, None, None], [4, 5, 6], [14, 15, 16], [1, 1, 1]],
    ],
)
def test_volumetric_index_overrider(
    override_offset,
    override_size,
    override_resolution,
    expected_start,
    expected_stop,
    expected_resolution,
):
    index = VolumetricIndex(resolution=Vec3D(1, 1, 1), bbox=BBox3D(((4, 14), (5, 15), (6, 16))))
    vio = VolumetricIndexOverrider(
        override_offset=override_offset,
        override_size=override_size,
        override_resolution=override_resolution,
    )
    index = vio(index)
    assert index.start == Vec3D(*expected_start)
    assert index.stop == Vec3D(*expected_stop)
    assert index.resolution == Vec3D(*expected_resolution)


@pytest.mark.parametrize(
    """res_change_mult,
    expected_start, expected_stop, expected_resolution""",
    [
        [[2, 2, 1], [2, 3, 6], [7, 7, 16], [2, 2, 1]],
        [[1, 2, 3], [4, 3, 2], [14, 7, 6], [1, 2, 3]],
    ],
)
def test_volumetric_index_scaler(
    res_change_mult,
    expected_start,
    expected_stop,
    expected_resolution,
):
    index = VolumetricIndex(resolution=Vec3D(1, 1, 1), bbox=BBox3D(((4, 14), (6, 14), (6, 16))))
    vis = VolumetricIndexScaler(res_change_mult=res_change_mult, allow_slice_rounding=True)
    index = vis(index)
    assert index.start == Vec3D(*expected_start)
    assert index.stop == Vec3D(*expected_stop)
    assert index.resolution == Vec3D(*expected_resolution)


def test_volumetric_index_scaler_error():
    index = VolumetricIndex(resolution=Vec3D(1, 1, 1), bbox=BBox3D(((4, 14), (6, 14), (6, 16))))
    vis = VolumetricIndexScaler(res_change_mult=[1, 2, 3])
    index = vis(index)
    with pytest.raises(ValueError):
        index.stop


@pytest.mark.parametrize(
    "start_coord, end_coord, resolution, targets, data_shape, expected_mask_region, existing_masks, upstream_pad",
    [
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            [],
            None,
        ),
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            ["target1"],
            None,
        ),
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2", "target3"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            ["target1", "target3"],
            None,
        ),
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2"],
            (1, 10, 10, 10),
            (slice(0, 5), slice(0, 5), slice(0, 5)),
            [],
            [0, 0, 0],
        ),
        (
            [0, 0, 0],
            [5, 5, 5],
            [1.0, 1.0, 1.0],
            ["target1", "target2"],
            (1, 10, 10, 10),
            (slice(1, 6), slice(2, 7), slice(3, 8)),
            [],
            [1, 2, 3],
        ),
    ],
)
def test_roi_mask_processor_read(
    start_coord,
    end_coord,
    resolution,
    targets,
    data_shape,
    expected_mask_region,
    existing_masks,
    upstream_pad,
):
    processor = ROIMaskProcessor(
        start_coord=start_coord,
        end_coord=end_coord,
        resolution=resolution,
        targets=targets,
        upstream_pad=upstream_pad,
    )

    idx = VolumetricIndex.from_coords(
        start_coord=Vec3D(*start_coord), end_coord=Vec3D(*end_coord), resolution=Vec3D(*resolution)
    )
    processor.process_index(idx, "read")

    data = {target: np.random.rand(*data_shape).astype(np.float32) for target in targets}
    for target in existing_masks:
        data[target + "_mask"] = np.ones(data_shape).astype(
            np.float32
        )  # Pre-existing mask for the target

    processed_data = processor.process_data(data, "read")

    for target in targets:
        assert target in processed_data

        if target in existing_masks:
            assert np.all(processed_data[target + "_mask"] == data[target + "_mask"])
        else:
            mask = processed_data[target + "_mask"]
            inside_roi = mask[
                0, expected_mask_region[0], expected_mask_region[1], expected_mask_region[2]
            ]
            assert np.all(inside_roi == 1)

            full_slice = slice(None)
            outside_roi_slices = tuple(
                full_slice if s == full_slice else slice(None, s.start)
                for s in expected_mask_region
            )
            outside_roi = mask[
                0, outside_roi_slices[0], outside_roi_slices[1], outside_roi_slices[2]
            ]
            assert np.all(outside_roi == 0)
