from zetta_utils.geometry import BBox3D
from zetta_utils.mazepa_layer_processing.common.interpolate_flow import build_interpolate_flow


def test_op_worker_type_is_plumbed_to_subchunkable(mocker):
    subchunkable_m = mocker.patch(
        "zetta_utils.mazepa_layer_processing.common.interpolate_flow."
        "build_subchunkable_apply_flow"
    )
    src = mocker.MagicMock()
    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(64, 64, 64))

    build_interpolate_flow(
        src=src,
        dst=None,
        src_resolution=(1.0, 1.0, 1.0),
        dst_resolutions=[(2.0, 2.0, 2.0), (4.0, 4.0, 4.0)],
        mode="img",
        processing_chunk_sizes=[[16, 16, 16]],
        bbox=bbox,
        op_worker_type="my-worker",
    )

    assert subchunkable_m.call_count == 2
    for call in subchunkable_m.call_args_list:
        assert call.kwargs["op_worker_type"] == "my-worker"


def test_op_worker_type_defaults_to_none(mocker):
    subchunkable_m = mocker.patch(
        "zetta_utils.mazepa_layer_processing.common.interpolate_flow."
        "build_subchunkable_apply_flow"
    )
    src = mocker.MagicMock()
    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(64, 64, 64))

    build_interpolate_flow(
        src=src,
        dst=None,
        src_resolution=(1.0, 1.0, 1.0),
        dst_resolutions=[(2.0, 2.0, 2.0)],
        mode="img",
        processing_chunk_sizes=[[16, 16, 16]],
        bbox=bbox,
    )

    assert subchunkable_m.call_count == 1
    assert subchunkable_m.call_args.kwargs["op_worker_type"] is None
