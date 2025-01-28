# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib

import numpy as np
import pytest
from cloudvolume.exceptions import ScaleUnavailableError

from zetta_utils.common import abspath
from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex
from zetta_utils.layer.volumetric.cloudvol.backend import CVBackend, _clear_cv_cache
from zetta_utils.layer.volumetric.precomputed import PrecomputedInfoSpec, precomputed

from ....helpers import assert_array_equal

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = THIS_DIR / "../../../assets/infos/"
LAYER_X0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_UINT63_0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_uint63_0")
LAYER_SCRATCH0_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_scratch0")

# Hack to mock a immutable method `write_info`
_write_info_notmock = precomputed._write_info


@pytest.fixture
def clear_caches_reset_mocks():
    _clear_cv_cache()
    precomputed._info_cache.clear()
    precomputed._write_info = _write_info_notmock


def test_cv_backend_bad_path_exc(clear_caches_reset_mocks):
    with pytest.raises(RuntimeError):
        CVBackend(path="abc")


def test_cv_backend_bad_scale_exc(clear_caches_reset_mocks):
    with pytest.raises(ScaleUnavailableError, match=abspath(LAYER_X0_PATH)):
        CVBackend(path=LAYER_X0_PATH).get_bounds(Vec3D(0, 0, 0))


def test_cv_backend_dtype(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(reference_path=LAYER_X0_PATH, data_type="uint8")
    cvb = CVBackend(path=LAYER_X0_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.dtype == np.dtype("uint8")


def test_cv_backend_info_expect_same_exc(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    with pytest.raises(RuntimeError):
        CVBackend(path=LAYER_X1_PATH, info_spec=info_spec, on_info_exists="expect_same")
    _write_info.assert_not_called()


def test_cv_backend_info_extend_exc_scales(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH, add_scales=[(4, 4, 1)], add_scales_mode="replace"
    )
    with pytest.raises(RuntimeError):
        CVBackend(path=LAYER_X0_PATH, info_spec=info_spec, on_info_exists="extend")
    _write_info.assert_not_called()


def test_cv_backend_info_extend_exc_nonscales(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(reference_path=LAYER_X0_PATH, data_type="float32")
    with pytest.raises(RuntimeError):
        CVBackend(path=LAYER_X0_PATH, info_spec=info_spec, on_info_exists="extend")
    _write_info.assert_not_called()


def test_cv_backend_info_extend(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(reference_path=LAYER_X0_PATH, add_scales=[[4, 4, 1]])
    CVBackend(path=LAYER_X0_PATH, info_spec=info_spec, on_info_exists="extend")
    _write_info.assert_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X0_PATH, LAYER_X0_PATH, "expect_same"],
        [LAYER_X0_PATH, None, "overwrite"],
        [LAYER_X0_PATH, None, "expect_same"],
    ],
)
def test_cv_backend_info_no_action(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists=mode)

    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "overwrite"],
    ],
)
def test_cv_backend_info_overwrite(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
        default_chunk_size=IntVec3D(999, 999, 1),
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists=mode)

    _write_info.assert_called_once()


def test_cv_backend_read(clear_caches_reset_mocks, mocker):
    data_read = np.ones([3, 4, 5, 2])
    expected = np.ones([2, 3, 4, 5])
    cv_m = mocker.MagicMock()
    cv_m.__getitem__ = mocker.MagicMock(return_value=data_read)
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cv_m.__getitem__ = mocker.MagicMock(return_value=data_read)
    cvb = CVBackend(path=LAYER_X0_PATH)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    result = cvb.read(index)
    assert_array_equal(result, expected)
    cv_m.__getitem__.assert_called_with(index.bbox.to_slices(index.resolution))


def test_cv_backend_write(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    cv_m.voxel_offset = [0, 1, 2]
    cv_m.chunk_size = [1, 1, 1]
    cv_m.volume_size = [16, 16, 16]
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    value = np.ones([2, 3, 4, 5])
    expected_written = np.ones([3, 4, 5, 2])  # channel as ch 0

    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    cvb.write(index, value)
    assert cv_m.__setitem__.call_args[0][0] == index.bbox.to_slices(index.resolution)
    assert_array_equal(
        cv_m.__setitem__.call_args[0][1],
        expected_written,
    )


def test_cv_backend_write_scalar(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    cv_m.voxel_offset = [0, 1, 2]
    cv_m.chunk_size = [1, 1, 1]
    cv_m.volume_size = [16, 16, 16]
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    value = np.array([1])
    expected_written = 1

    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    cvb.write(index, value)

    assert cv_m.__setitem__.call_args[0][0] == index.bbox.to_slices(index.resolution)
    assert cv_m.__setitem__.call_args[0][1] == expected_written


def test_cv_backend_read_uint63(clear_caches_reset_mocks, mocker):
    data_read = np.array([[[[2 ** 63 - 1]]]], dtype=np.uint64)
    expected = np.array([[[[2 ** 63 - 1]]]], dtype=np.int64)
    cv_m = mocker.MagicMock()
    cv_m.__getitem__ = mocker.MagicMock(return_value=data_read)
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_UINT63_0_PATH)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(0, 1), slice(0, 1))),
        resolution=Vec3D(1, 1, 1),
    )
    result = cvb.read(index)
    assert_array_equal(result, expected)
    cv_m.__getitem__.assert_called_with(index.bbox.to_slices(index.resolution))


def test_cv_backend_write_scalar_uint63(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_UINT63_0_PATH,
    )
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    cv_m.voxel_offset = [0, 0, 0]
    cv_m.chunk_size = [1, 1, 1]
    cv_m.volume_size = [16, 16, 16]
    cv_m.dtype = "uint64"
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    value = np.array([2 ** 63 - 1], dtype=np.int64)
    expected_written = np.uint64(2 ** 63 - 1)

    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(0, 1), slice(0, 1))),
        resolution=Vec3D(1, 1, 1),
    )
    cvb.write(index, value)
    assert cv_m.__setitem__.call_args[0][0] == index.bbox.to_slices(index.resolution)
    assert cv_m.__setitem__.call_args[0][1] == expected_written
    assert cv_m.__setitem__.call_args[0][1].dtype == expected_written.dtype


def test_cv_backend_write_scalar_uint63_exc(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_UINT63_0_PATH,
    )
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    cv_m.voxel_offset = [0, 0, 0]
    cv_m.chunk_size = [1, 1, 1]
    cv_m.volume_size = [16, 16, 16]
    cv_m.dtype = "uint64"
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    value = np.array([-1], dtype=np.int64)

    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(0, 1), slice(0, 1))),
        resolution=Vec3D(1, 1, 1),
    )
    with pytest.raises(ValueError):
        cvb.write(index, value)


@pytest.mark.parametrize(
    "data_in,expected_exc",
    [
        # Too many dims
        [np.ones((1, 2, 3, 4, 5, 6)), ValueError],
    ],
)
def test_cv_backend_write_exc(clear_caches_reset_mocks, data_in, expected_exc, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    cv_m.voxel_offset = [0, 0, 0]
    cv_m.chunk_size = [1, 1, 1]
    cv_m.volume_size = [16, 16, 16]
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(1, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    with pytest.raises(expected_exc):
        cvb.write(index, data_in)


def test_cv_get_chunk_size(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.get_chunk_size(Vec3D(2, 2, 1)) == IntVec3D(1024, 1024, 1)


def test_cv_get_voxel_offset(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.get_voxel_offset(Vec3D(2, 2, 1)) == IntVec3D(1, 2, 3)


def test_cv_get_dataset_size(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_dataset_size=IntVec3D(4096, 4096, 1),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.get_dataset_size(Vec3D(2, 2, 1)) == IntVec3D(4096, 4096, 1)


def test_cv_with_changes(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    cvb_new = cvb.with_changes(
        name=LAYER_SCRATCH0_PATH,
        voxel_offset_res=(IntVec3D(3, 2, 1), Vec3D(2, 2, 1)),
        chunk_size_res=(IntVec3D(512, 512, 1), Vec3D(2, 2, 1)),
        dataset_size_res=(IntVec3D(2048, 2048, 1), Vec3D(2, 2, 1)),
        enforce_chunk_aligned_writes=False,
    )
    assert cvb_new.name == LAYER_SCRATCH0_PATH
    assert cvb_new.get_voxel_offset(Vec3D(2, 2, 1)) == IntVec3D(3, 2, 1)
    assert cvb_new.get_chunk_size(Vec3D(2, 2, 1)) == IntVec3D(512, 512, 1)
    assert cvb_new.get_dataset_size(Vec3D(2, 2, 1)) == IntVec3D(2048, 2048, 1)
    assert not cvb_new.enforce_chunk_aligned_writes


def test_cv_with_changes_exc(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    with pytest.raises(KeyError):
        cvb.with_changes(nonsensename="nonsensevalue")


def test_cv_assert_idx_is_chunk_aligned(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 4), slice(-8, 12), slice(-18, -11)), resolution=Vec3D(2, 2, 1)
        ),
        resolution=Vec3D(2, 2, 1),
    )
    cvb.assert_idx_is_chunk_aligned(index)


def test_cv_assert_idx_is_chunk_aligned_crop(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 16384), slice(2, 16387), slice(3, 16390)), resolution=Vec3D(2, 2, 1)
        ),
        resolution=Vec3D(2, 2, 1),
    )
    cvb.assert_idx_is_chunk_aligned(index)


def test_cv_assert_idx_is_chunk_aligned_exc(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(0, 13), slice(0, 13), slice(0, 13)), resolution=Vec3D(2, 2, 1)
        ),
        resolution=Vec3D(2, 2, 1),
    )

    with pytest.raises(ValueError):
        cvb.assert_idx_is_chunk_aligned(index)


def test_cv_assert_idx_is_chunk_aligned_crop_exc(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 16384), slice(2, 16386), slice(3, -11)), resolution=Vec3D(2, 2, 1)
        ),
        resolution=Vec3D(2, 2, 1),
    )

    with pytest.raises(ValueError):
        cvb.assert_idx_is_chunk_aligned(index)


def test_cv_assert_idx_is_chunk_aligned_crop_preorigin_exc(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_SCRATCH0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 16384), slice(2, 16386), slice(0, 16387)), resolution=Vec3D(2, 2, 1)
        ),
        resolution=Vec3D(2, 2, 1),
    )

    with pytest.raises(ValueError):
        cvb.assert_idx_is_chunk_aligned(index)
