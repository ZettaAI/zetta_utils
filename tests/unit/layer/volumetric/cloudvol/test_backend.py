# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib

import pytest
import torch

from zetta_utils.geometry import BBox3D, IntVec3D, Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, cloudvol
from zetta_utils.layer.volumetric.cloudvol import CVBackend

from ....helpers import assert_array_equal

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = THIS_DIR / "../../../assets/infos/"
LAYER_X0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_X2_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x2")
LAYER_X3_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x3")
LAYER_X4_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x4")
LAYER_X5_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x5")


@pytest.fixture
def clear_caches():
    cloudvol.backend._info_cache.clear()


def test_cv_backend_specific_mip_exc():
    with pytest.raises(ValueError):
        CVBackend(path="", cv_kwargs={"mip": 4})


def test_cv_backend_bad_path_exc():
    with pytest.raises(FileNotFoundError):
        CVBackend(path="abc")


def test_cv_backend_dtype():
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH, data_type="uint8"
    )
    cvb = CVBackend(path=LAYER_X0_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.dtype == torch.uint8


def test_cv_backend_dtype_exc():
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH, data_type="nonsense"
    )
    cvb = CVBackend(path=LAYER_X0_PATH, info_spec=info_spec, on_info_exists="overwrite")
    with pytest.raises(ValueError):
        cvb.dtype


def test_cv_backend_info_expect_same_exc(mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()

    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    with pytest.raises(RuntimeError):
        CVBackend(path=LAYER_X1_PATH, info_spec=info_spec, on_info_exists="expect_same")
    cv_m.commit_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X0_PATH, LAYER_X0_PATH, "expect_same"],
        [LAYER_X0_PATH, None, "overwrite"],
        [LAYER_X0_PATH, None, "expect_same"],
    ],
)
def test_cv_backend_info_no_action(path, reference, mode, mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=reference,
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists=mode)

    cv_m.commit_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "overwrite"],
    ],
)
def test_cv_backend_info_overwrite(clear_caches, path, reference, mode, mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=reference,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists=mode)

    cv_m.commit_info.assert_called_once()


@pytest.mark.parametrize(
    "path, reference",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH],
    ],
)
def test_cv_backend_info_no_overwrite_when_same_as_cached(clear_caches, path, reference, mocker):
    cv_m = mocker.MagicMock()
    mocker.patch(
        "cloudvolume.CloudVolume.__new__",
        return_value=cv_m,
    )
    cv_m.commit_info = mocker.MagicMock()
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=reference,
    )
    CVBackend(path=path, info_spec=info_spec, on_info_exists="overwrite")

    cv_m.commit_info.assert_not_called()


def test_cv_backend_read(clear_caches, mocker):
    data_read = torch.ones([3, 4, 5, 2])
    expected = torch.ones([2, 3, 4, 5])
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


def test_cv_backend_write(clear_caches, mocker):
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_X0_PATH)
    value = torch.ones([2, 3, 4, 5])
    expected_written = torch.ones([3, 4, 5, 2])  # channel as ch 0

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


def test_cv_backend_write_scalar(clear_caches, mocker):
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_X0_PATH)
    value = torch.tensor([1])
    expected_written = 1

    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(0, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    cvb.write(index, value)
    assert cv_m.__setitem__.call_args[0][0] == index.bbox.to_slices(index.resolution)
    assert cv_m.__setitem__.call_args[0][1] == expected_written


@pytest.mark.parametrize(
    "data_in,expected_exc",
    [
        # Too many dims
        [torch.ones((1, 2, 3, 4, 5, 6)), ValueError],
    ],
)
def test_cv_backend_write_exc(data_in, expected_exc, clear_caches, mocker):
    cv_m = mocker.MagicMock()
    cv_m.__setitem__ = mocker.MagicMock()
    mocker.patch("cloudvolume.CloudVolume.__new__", return_value=cv_m)
    cvb = CVBackend(path=LAYER_X0_PATH)
    index = VolumetricIndex(
        bbox=BBox3D.from_slices((slice(1, 1), slice(1, 2), slice(2, 3))),
        resolution=Vec3D(1, 1, 1),
    )
    with pytest.raises(expected_exc):
        cvb.write(index, data_in)


def test_cv_get_chunk_size(clear_caches):
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    cvb = CVBackend(path=LAYER_X2_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.get_chunk_size(Vec3D(8, 8, 8)) == IntVec3D(1024, 1024, 1)


def test_cv_get_voxel_offset(clear_caches):
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_X3_PATH, info_spec=info_spec, on_info_exists="overwrite")

    assert cvb.get_voxel_offset(Vec3D(8, 8, 8)) == IntVec3D(1, 2, 3)


def test_cv_with_changes(clear_caches, mocker):
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    cvb = CVBackend(path=LAYER_X5_PATH, info_spec=info_spec, on_info_exists="overwrite")
    cvb_new = cvb.with_changes(
        name=LAYER_X5_PATH,
        voxel_offset_res=(IntVec3D(3, 2, 1), Vec3D(8, 8, 8)),
        chunk_size_res=(IntVec3D(512, 512, 1), Vec3D(8, 8, 8)),
        use_compression=False,
        enforce_chunk_aligned_writes=False,
    )
    assert cvb_new.name == LAYER_X5_PATH
    assert cvb_new.get_voxel_offset(Vec3D(8, 8, 8)) == IntVec3D(3, 2, 1)
    assert cvb_new.get_chunk_size(Vec3D(8, 8, 8)) == IntVec3D(512, 512, 1)
    assert not cvb_new.use_compression
    assert not cvb_new.enforce_chunk_aligned_writes


def test_cv_with_changes_exc(clear_caches, mocker):
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X5_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    cvb = CVBackend(path=LAYER_X5_PATH, info_spec=info_spec, on_info_exists="overwrite")
    with pytest.raises(KeyError):
        cvb.with_changes(nonsensename="nonsensevalue")


def test_cv_assert_idx_is_chunk_aligned(clear_caches):
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_X4_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(1, 4), slice(-8, 12), slice(-18, -11)), resolution=Vec3D(8, 8, 8)
        ),
        resolution=Vec3D(8, 8, 8),
    )
    cvb.assert_idx_is_chunk_aligned(index)


def test_cv_assert_idx_is_chunk_aligned_exc(clear_caches):
    info_spec = cloudvol.backend.PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(3, 5, 7),
        default_voxel_offset=IntVec3D(1, 2, 3),
    )
    cvb = CVBackend(path=LAYER_X4_PATH, info_spec=info_spec, on_info_exists="overwrite")
    index = VolumetricIndex(
        bbox=BBox3D.from_slices(
            (slice(0, 13), slice(0, 13), slice(0, 13)), resolution=Vec3D(8, 8, 8)
        ),
        resolution=Vec3D(8, 8, 8),
    )

    with pytest.raises(ValueError):
        cvb.assert_idx_is_chunk_aligned(index)
