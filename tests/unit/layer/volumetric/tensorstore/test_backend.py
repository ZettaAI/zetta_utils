# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib
from copy import deepcopy

import numpy as np
import pytest
import tensorstore

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer import precomputed
from zetta_utils.layer.precomputed import InfoSpecParams, PrecomputedInfoSpec
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.cloudvol.backend import CVBackend
from zetta_utils.layer.volumetric.layer_set import VolumetricSetBackend
from zetta_utils.layer.volumetric.tensorstore.backend import TSBackend, _clear_ts_cache

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = os.path.abspath(THIS_DIR / "../../../assets/infos/")
LAYER_X0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_SCRATCH0_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x0")
LAYER_SCRATCH1_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x1")

# Hack to mock a immutable method `write_info`
_write_info_notmock = precomputed._write_info
_TensorStore_notmock = deepcopy(tensorstore.TensorStore)


@pytest.fixture
def clear_caches_reset_mocks():
    _clear_ts_cache()
    precomputed._info_cache.clear()
    precomputed._write_info = _write_info_notmock
    tensorstore.TensorStore = deepcopy(_TensorStore_notmock)


def test_ts_backend_bad_path_exc(clear_caches_reset_mocks):
    with pytest.raises(Exception):
        TSBackend(path="abc")


def test_ts_backend_dtype(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            data_type="uint8",
            scales=[[1, 1, 1]],
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_X0_PATH, info_spec=info_spec, info_overwrite=True)
    assert tsb.dtype == np.dtype("uint8")


def test_ts_backend_info_expect_same_exc(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[8, 8, 8]],
            chunk_size=[3, 3, 3],
            inherit_all_params=True,
        )
    )
    with pytest.raises(Exception):
        TSBackend(path=LAYER_X1_PATH, info_spec=info_spec, info_overwrite=False)
    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, overwrite",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, True],
        [LAYER_X0_PATH, LAYER_X0_PATH, False],
    ],
)
def test_ts_backend_info_no_action(clear_caches_reset_mocks, path, reference, overwrite, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=reference,
            scales=[[1, 1, 1]],
            inherit_all_params=True,
        )
    )
    TSBackend(path=path, info_spec=info_spec, info_overwrite=overwrite)

    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference",
    [
        [
            LAYER_SCRATCH0_PATH,
            LAYER_X0_PATH,
        ],
        [".", LAYER_X0_PATH],
    ],
)
def test_ts_backend_info_overwrite(clear_caches_reset_mocks, path, reference, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=reference,
            scales=[[1, 1, 1]],
            chunk_size=[999, 999, 1],
            inherit_all_params=True,
        )
    )
    TSBackend(path=path, info_spec=info_spec, info_overwrite=True)

    _write_info.assert_called_once()


@pytest.mark.parametrize(
    "path, reference",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH],
    ],
)
def test_ts_backend_info_no_overwrite_when_same_as_cached(
    clear_caches_reset_mocks, path, reference, mocker
):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=reference,
            scales=[[1, 1, 1]],
            inherit_all_params=True,
        )
    )
    TSBackend(path=path, info_spec=info_spec, info_overwrite=True)

    _write_info.assert_not_called()


def test_ts_backend_read_idx(clear_caches_reset_mocks, mocker):
    expected_shape = (1, 3, 4, 5)
    tsb = TSBackend(path=LAYER_X0_PATH)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(0, 3), slice(1, 5), slice(2, 7)), resolution=Vec3D(1, 1, 1)
        ),
        resolution=Vec3D(1, 1, 1),
    )
    result = tsb.read(index)
    assert result.shape == expected_shape


def test_ts_backend_read_partial(clear_caches_reset_mocks, mocker):
    tensorstore.TensorStore.__getitem__ = mocker.MagicMock(
        return_value=np.ones(shape=(1, 1, 1, 1), dtype=np.uint8)
    )
    tsb = TSBackend(path=LAYER_X0_PATH)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(-1, 1), slice(0, 1), slice(0, 1)),
            resolution=Vec3D(1, 1, 1),
        ),
        resolution=Vec3D(1, 1, 1),
    )
    result = tsb.read(index)
    assert result[:, 0:1, :, :] == np.zeros((1, 1, 1, 1), dtype=np.uint8)
    assert result[:, 1:2, :, :] == np.ones((1, 1, 1, 1), dtype=np.uint8)


def test_ts_backend_write_idx(clear_caches_reset_mocks, mocker):
    tensorstore.TensorStore.__setitem__ = mocker.MagicMock()
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[3, 5, 7],
            bbox=BBox3D.from_coords(
                start_coord=[1, 2, 3], end_coord=[101, 102, 103], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    value = np.ones([1, 3, 5, 7], dtype=np.uint8)

    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 4), slice(2, 7), slice(3, 10)),
            resolution=Vec3D(1, 1, 1),
        ),
        resolution=Vec3D(1, 1, 1),
    )
    tsb.write(index, value)
    tensorstore.TensorStore.__setitem__.call_args_list[0] = mocker.call(
        (slice(1, 4), slice(2, 7), slice(3, 10)), np.ones((3, 5, 7, 1), dtype=np.uint8)
    )


def test_ts_backend_write_idx_partial(clear_caches_reset_mocks, mocker):
    tensorstore.TensorStore.__setitem__ = mocker.MagicMock()
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[4, 4, 1],
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[100, 100, 100], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(
        path=LAYER_SCRATCH0_PATH,
        info_spec=info_spec,
        info_overwrite=True,
        enforce_chunk_aligned_writes=False,
    )
    value = np.ones([1, 3, 4, 5], dtype=np.uint8)

    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(-2, 1), slice(-3, 1), slice(-4, 1)),
            resolution=Vec3D(1, 1, 1),
        ),
        resolution=Vec3D(1, 1, 1),
    )
    tsb.write(index, value)
    tensorstore.TensorStore.__setitem__.assert_called_with(
        (slice(0, 1), slice(0, 1), slice(0, 1)), np.array([[[[1]]]], dtype=np.uint8)
    )


def test_ts_backend_write_scalar_idx(clear_caches_reset_mocks, mocker):
    tensorstore.TensorStore.__setitem__ = mocker.MagicMock()
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[3, 5, 7],
            bbox=BBox3D.from_coords(
                start_coord=[1, 2, 3], end_coord=[101, 102, 103], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    value = np.array([1], dtype=np.uint8)

    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 4), slice(2, 7), slice(3, 10)),
            resolution=Vec3D(1, 1, 1),
        ),
        resolution=Vec3D(1, 1, 1),
    )
    tsb.write(index, value)
    tensorstore.TensorStore.__setitem__.assert_called_with(
        (slice(1, 4), slice(2, 7), slice(3, 10)), np.array([[[[1]]]], dtype=np.uint8)
    )


@pytest.mark.parametrize(
    "data_in,expected_exc",
    [
        # Too many dims
        [np.ones((1, 2, 3, 4, 5, 6)), ValueError],
    ],
)
def test_ts_backend_write_exc_dims(data_in, expected_exc, clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1, 1, 1],
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[100, 100, 100], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(1, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    with pytest.raises(expected_exc):
        tsb.write(index, data_in)


@pytest.mark.parametrize(
    "data_in,expected_exc",
    [
        # idx not chunk aligned
        [np.ones((3, 3, 3, 3), dtype=np.uint8), ValueError],
    ],
)
def test_ts_backend_write_exc_chunks(data_in, expected_exc, clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[5, 5, 5],
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[100, 100, 100], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 3), slice(0, 3), slice(0, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    with pytest.raises(expected_exc):
        tsb.write(index, data_in)


def test_ts_get_chunk_size(clear_caches_reset_mocks):
    precomputed._write_info = _write_info_notmock
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1024, 1024, 1],
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)

    assert tsb.get_chunk_size(Vec3D(1, 1, 1)) == IntVec3D(1024, 1024, 1)


def test_ts_get_voxel_offset(clear_caches_reset_mocks):
    precomputed._write_info = _write_info_notmock
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1024, 1024, 1],
            bbox=BBox3D.from_coords(
                start_coord=[1, 2, 3], end_coord=[1025, 1026, 4], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)

    assert tsb.get_voxel_offset(Vec3D(1, 1, 1)) == IntVec3D(1, 2, 3)


def test_ts_get_dataset_size(clear_caches_reset_mocks):
    precomputed._write_info = _write_info_notmock
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1024, 1024, 1],
            bbox=BBox3D.from_coords(
                start_coord=[0, 0, 0], end_coord=[4096, 4096, 1], resolution=[1, 1, 1]
            ),
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)

    assert tsb.get_dataset_size(Vec3D(1, 1, 1)) == IntVec3D(4096, 4096, 1)


def test_ts_with_changes(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1024, 1024, 1],
            inherit_all_params=True,
        )
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    tsb_new = tsb.with_changes(
        name=LAYER_SCRATCH1_PATH,
        voxel_offset_res=(IntVec3D(3, 2, 1), Vec3D(1, 1, 1)),
        chunk_size_res=(IntVec3D(512, 512, 1), Vec3D(1, 1, 1)),
        dataset_size_res=(IntVec3D(2048, 2048, 1), Vec3D(1, 1, 1)),
        enforce_chunk_aligned_writes=False,
    )
    assert tsb_new.name == LAYER_SCRATCH1_PATH
    assert tsb_new.get_voxel_offset(Vec3D(1, 1, 1)) == IntVec3D(3, 2, 1)
    assert tsb_new.get_chunk_size(Vec3D(1, 1, 1)) == IntVec3D(512, 512, 1)
    assert tsb_new.get_dataset_size(Vec3D(1, 1, 1)) == IntVec3D(2048, 2048, 1)
    assert not tsb_new.enforce_chunk_aligned_writes


def test_ts_from_precomputed_cv(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1024, 1024, 1],
            inherit_all_params=True,
        )
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    tsb = TSBackend.from_precomputed(cvb)
    assert tsb.path == cvb.path
    assert tsb.info_spec == cvb.info_spec
    assert tsb.info_overwrite == cvb.info_overwrite


def test_ts_from_precomputed_layer_set(clear_caches_reset_mocks, mocker):
    info_spec1 = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1024, 1024, 1],
            inherit_all_params=True,
        )
    )
    info_spec2 = PrecomputedInfoSpec(
        info_spec_params=InfoSpecParams.from_optional_reference(
            reference_path=LAYER_X0_PATH,
            scales=[[1, 1, 1]],
            chunk_size=[1023, 1023, 1],
            inherit_all_params=True,
        )
    )
    cvb1 = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec1, info_overwrite=True)
    cvb2 = CVBackend(path=LAYER_SCRATCH1_PATH, info_spec=info_spec2, info_overwrite=True)
    lsb = VolumetricSetBackend(
        layers={
            "cvb1": VolumetricLayer(backend=cvb1),
            "cvb2": VolumetricLayer(backend=cvb2),
        }
    )
    tsb = TSBackend.from_precomputed(lsb)
    assert TSBackend.from_precomputed(lsb.layers["cvb1"].backend) == tsb.layers["cvb1"].backend
    assert TSBackend.from_precomputed(lsb.layers["cvb2"].backend) == tsb.layers["cvb2"].backend


def test_ts_with_changes_exc(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        info_path=LAYER_X0_PATH,
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    with pytest.raises(KeyError):
        tsb.with_changes(nonsensename="nonsensevalue")


def test_ts_assert_idx_is_chunk_aligned(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_path=LAYER_X0_PATH,
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(0, 1024), slice(-1024, 1024), slice(1024, 2048)), resolution=Vec3D(1, 1, 1)
        ),
        resolution=Vec3D(1, 1, 1),
    )
    tsb.assert_idx_is_chunk_aligned(index)


def test_ts_assert_idx_is_chunk_aligned_crop(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_path=LAYER_X0_PATH,
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(0, 16384), slice(0, 16384), slice(3, 16390)), resolution=Vec3D(1, 1, 1)
        ),
        resolution=Vec3D(1, 1, 1),
    )
    tsb.assert_idx_is_chunk_aligned(index)


def test_ts_assert_idx_is_chunk_aligned_exc(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_path=LAYER_X0_PATH,
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 5), slice(-8, 12), slice(-18, -11)), resolution=Vec3D(1, 1, 1)
        ),
        resolution=Vec3D(1, 1, 1),
    )

    with pytest.raises(ValueError):
        tsb.assert_idx_is_chunk_aligned(index)


def test_ts_assert_idx_is_chunk_crop_aligned_exc(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        info_path=LAYER_X0_PATH,
    )
    tsb = TSBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, info_overwrite=True)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 16384), slice(2, 16386), slice(0, 16387)), resolution=Vec3D(1, 1, 1)
        ),
        resolution=Vec3D(1, 1, 1),
    )

    with pytest.raises(ValueError):
        tsb.assert_idx_is_chunk_aligned(index)
