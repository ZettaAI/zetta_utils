from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.mazepa_layer_processing.common.volumetric_apply_flow import (
    VolumetricApplyFlowSchema,
)


def test_get_temp_dst_sizes_from_idx_shape_not_full_dst(mocker):
    op = mocker.MagicMock()
    schema: VolumetricApplyFlowSchema = VolumetricApplyFlowSchema(
        op=op,
        processing_chunk_size=Vec3D[int](4, 4, 4),
        dst_resolution=Vec3D(1.0, 1.0, 1.0),
        intermediaries_dir="file:///tmp/intermediaries",
    )

    bbox = BBox3D.from_coords(start_coord=(0, 0, 0), end_coord=(8, 8, 8))
    idx = VolumetricIndex(bbox=bbox, resolution=Vec3D(1.0, 1.0, 1.0))

    new_backend = mocker.MagicMock()
    backend = mocker.MagicMock()
    backend.with_changes = mocker.MagicMock(return_value=new_backend)
    backend.get_dataset_size = mocker.MagicMock(return_value=Vec3D[int](1000, 1000, 1000))
    dst = mocker.MagicMock()
    dst.backend = backend

    schema._get_temp_dst(  # pylint: disable=protected-access
        dst=dst, idx=idx, prefix="pre", suffix="suf"
    )

    backend.get_dataset_size.assert_not_called()
    kwargs = backend.with_changes.call_args.kwargs
    expected_size = idx.shape + 2 * schema.processing_chunk_size
    assert kwargs["dataset_size_res"] == (expected_size, schema.dst_resolution)
