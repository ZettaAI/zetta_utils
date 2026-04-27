from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow import (
    _expand_bbox_processing,
    _expand_bbox_resolution,
)


def test_expand_bbox_resolution_logs_when_already_aligned_and_verbose(mocker):
    log_m = mocker.patch(
        "zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow.logger.info"
    )
    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(10, 10, 10))
    dst_resolution = Vec3D(1.0, 1.0, 1.0)

    result = _expand_bbox_resolution(bbox, dst_resolution, verbose=True)

    assert result == bbox
    assert log_m.call_count == 1


def test_expand_bbox_resolution_silent_when_already_aligned_and_not_verbose(mocker):
    log_m = mocker.patch(
        "zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow.logger.info"
    )
    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(10, 10, 10))
    dst_resolution = Vec3D(1.0, 1.0, 1.0)

    result = _expand_bbox_resolution(bbox, dst_resolution, verbose=False)

    assert result == bbox
    log_m.assert_not_called()


def test_expand_bbox_processing_silent_when_realigned_and_not_verbose(mocker):
    log_m = mocker.patch(
        "zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow.logger.info"
    )
    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(7, 7, 7))
    dst_resolution = Vec3D(1.0, 1.0, 1.0)
    processing_chunk_sizes = [Vec3D[int](4, 4, 4)]
    processing_gap = Vec3D[int](0, 0, 0)

    result = _expand_bbox_processing(
        bbox, dst_resolution, processing_chunk_sizes, processing_gap, verbose=False
    )

    assert result != bbox
    log_m.assert_not_called()


def test_expand_bbox_processing_logs_when_realigned_and_verbose(mocker):
    log_m = mocker.patch(
        "zetta_utils.mazepa_layer_processing.common.subchunkable_apply_flow.logger.info"
    )
    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(7, 7, 7))
    dst_resolution = Vec3D(1.0, 1.0, 1.0)
    processing_chunk_sizes = [Vec3D[int](4, 4, 4)]
    processing_gap = Vec3D[int](0, 0, 0)

    _expand_bbox_processing(
        bbox, dst_resolution, processing_chunk_sizes, processing_gap, verbose=True
    )

    assert log_m.call_count == 1
