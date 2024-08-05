# pylint: disable=missing-docstring,redefined-outer-name,unused-argument,pointless-statement,line-too-long,protected-access,too-few-public-methods
import os
import pathlib

import pytest

from zetta_utils.geometry import IntVec3D, Vec3D
from zetta_utils.layer.volumetric.precomputed import (
    PrecomputedInfoSpec,
    get_info,
    precomputed,
)

THIS_DIR = pathlib.Path(__file__).parent.resolve()
INFOS_DIR = os.path.abspath(THIS_DIR / "../../../assets/infos/")
LAYER_X0_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x0")
LAYER_X1_PATH = "file://" + os.path.join(INFOS_DIR, "layer_x1")
LAYER_SCRATCH0_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "layer_x0")
NONEXISTENT_LAYER_PATH = "file://" + os.path.join(INFOS_DIR, "scratch", "nonexistent_layer")

EXAMPLE_INFO = '{"data_type": "uint8", "num_channels": 1, "scales": [{"chunk_sizes": [[1024, 1024, 1]], "encoding": "raw", "key": "4_4_40", "resolution": [4, 4, 40], "size": [16384, 16384, 16384], "voxel_offset": [0, 0, 0]}, {"chunk_sizes": [[2048, 2048, 1]], "encoding": "raw", "key": "8_8_40", "resolution": [8, 8, 40], "size": [8192, 8192, 8192], "voxel_offset": [0, 0, 0]}], "type": "raw"}'

_write_info_notmock = precomputed._write_info


@pytest.fixture
def clear_caches_reset_mocks():
    precomputed._info_cache.clear()
    precomputed._write_info = _write_info_notmock


def test_nonexistent_info(clear_caches_reset_mocks):
    with pytest.raises(FileNotFoundError):
        precomputed.get_info(NONEXISTENT_LAYER_PATH)


def test_infospec_expect_same_exc(clear_caches_reset_mocks, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(path=LAYER_X1_PATH, on_info_exists="expect_same")
    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X0_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X0_PATH, LAYER_X0_PATH, "expect_same"],
        [LAYER_X0_PATH, None, "overwrite"],
        [LAYER_X0_PATH, None, "expect_same"],
    ],
)
def test_infospec_no_action(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    _write_info.assert_not_called()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [".", LAYER_X0_PATH, "overwrite"],
    ],
)
def test_infospec_overwrite(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
        default_chunk_size=IntVec3D(999, 999, 1),
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    _write_info.assert_called_once()


@pytest.mark.parametrize(
    "path, reference, mode",
    [
        [LAYER_X1_PATH, LAYER_X0_PATH, "overwrite"],
        [LAYER_X1_PATH + "yo", LAYER_X0_PATH, "overwrite"],
    ],
)
def test_infospec_extend(clear_caches_reset_mocks, path, reference, mode, mocker):
    _write_info = mocker.MagicMock()
    precomputed._write_info = _write_info
    info_spec = PrecomputedInfoSpec(
        reference_path=reference,
        extend_if_exists_path=path,
        default_chunk_size=IntVec3D(999, 999, 1),
        add_scales=[[128, 128, 1]],
    )
    info_spec.update_info(path=path, on_info_exists=mode)

    _write_info.assert_called_once()


def test_exclude_fields(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH, add_scales=[[1, 1, 1]], add_scales_exclude_fields=["trash"]
    )
    info = info_spec.make_info()
    assert info is not None
    assert "trash" not in info["scales"][0]


def test_type(clear_caches_reset_mocks):
    info_spec = PrecomputedInfoSpec(reference_path=LAYER_X0_PATH, type="segmentation")
    info = info_spec.make_info()
    assert info is not None
    assert info["type"] == "segmentation"


def test_set_voxel_offset_chunk_and_data(clear_caches_reset_mocks, mocker):
    info_spec = PrecomputedInfoSpec(
        reference_path=LAYER_X0_PATH,
        default_chunk_size=IntVec3D(1024, 1024, 1),
    )
    info_spec.add_scales = [[2, 2, 1]]
    info_spec.set_chunk_size((IntVec3D(3, 2, 1), Vec3D(2, 2, 1)))
    info_spec.set_voxel_offset((IntVec3D(1, 2, 3), Vec3D(2, 2, 1)))
    info_spec.set_dataset_size((IntVec3D(10, 20, 30), Vec3D(2, 2, 1)))
    info_spec.update_info(LAYER_SCRATCH0_PATH, on_info_exists="overwrite")
    scales = get_info(LAYER_SCRATCH0_PATH)["scales"]
    for scale in scales:
        if scale["resolution"] == [2, 2, 1]:
            assert scale["voxel_offset"] == [1, 2, 3]
            assert scale["chunk_sizes"][0] == [3, 2, 1]
            assert scale["size"] == [10, 20, 30]


def make_tmp_layer(tmpdir, info_text=None):
    if info_text is None:
        info_text = EXAMPLE_INFO
    tmpdir_info = tmpdir.join("info")
    tmpdir_info.write(info_text)
    return str(tmpdir)


@pytest.mark.parametrize(
    "in_scales, expects",
    [
        # test shorthand
        [[[16, 16, 40]], ["16_16_40"]],
        # test multiples
        [[[16, 16, 40], [32, 32, 40]], ["16_16_40", "32_32_40"]],
        # test minimum spec of just `resolution`
        [[{"resolution": [16, 16, 40]}], ["16_16_40"]],
        # test mixed
        [[{"resolution": [16, 16, 40]}, [32, 32, 40]], ["16_16_40", "32_32_40"]],
    ],
)
def test_add_scale(clear_caches_reset_mocks, tmpdir, in_scales, expects):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=in_scales,
    )
    info_spec.update_info(tmp_layer, on_info_exists="overwrite")
    scales = get_info(tmp_layer)["scales"]
    for expect in expects:
        assert expect in [e["key"] for e in scales]


@pytest.mark.parametrize(
    "mode, expects",
    [
        ["merge", ["4_4_40", "8_8_40", "16_16_40"]],
        ["replace", ["16_16_40"]],
    ],
)
def test_add_scale_modes(clear_caches_reset_mocks, tmpdir, mode, expects):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=[[16, 16, 40]],
        add_scales_mode=mode,
    )
    info_spec.update_info(tmp_layer, on_info_exists="overwrite")
    scales = get_info(tmp_layer)["scales"]
    for expect in expects:
        assert expect in [e["key"] for e in scales]


def test_add_scale_full_entry(clear_caches_reset_mocks, tmpdir):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=[
            {
                "key": "33_33_40",
                "resolution": [33, 33, 40],
                "chunk_sizes": [[7, 8, 9]],
                "encoding": "raw",
                "size": [1000, 1000, 1000],
                "voxel_offset": [4, 5, 6],
            },
        ],
    )
    info_spec.update_info(tmp_layer, on_info_exists="overwrite")
    scales = get_info(tmp_layer)["scales"]
    assert "33_33_40" in [e["key"] for e in scales]
    for scale in scales:
        if scale["key"] == "33_33_40":
            assert scale["chunk_sizes"] == [[7, 8, 9]]


def test_add_scale_unsupported_keys(clear_caches_reset_mocks, tmpdir):
    """Throw error if key does not match resolution"""
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=[
            {
                "key": "40_33_33",
                "resolution": [33, 33, 40],
                "chunk_sizes": [[7, 8, 9]],
                "encoding": "raw",
                "size": [1000, 1000, 1000],
                "voxel_offset": [4, 5, 6],
            },
        ],
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(tmp_layer, on_info_exists="overwrite")


def test_add_scale_with_ref(clear_caches_reset_mocks, tmpdir):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales_ref="8_8_40",
        add_scales=[[16, 16, 40]],
    )
    info_spec.update_info(tmp_layer, on_info_exists="overwrite")
    scales = get_info(tmp_layer)["scales"]
    assert "16_16_40" in [e["key"] for e in scales]


# def test_add_scale_error_non_multiple(clear_caches_reset_mocks, tmpdir):
#     tmp_layer = make_tmp_layer(tmpdir)
#     info_spec = PrecomputedInfoSpec(
#         reference_path=tmp_layer,
#         add_scales=[[6, 6, 82]],
#     )
#     with pytest.raises(RuntimeError):
#         breakpoint()
# info_spec.update_info(tmp_layer, on_info_exists="overwrite")


def test_add_scale_error_no_ref_scales(clear_caches_reset_mocks, tmpdir):
    info = '{"data_type": "uint8", "num_channels": 1, "type": "raw"}'
    tmp_layer = make_tmp_layer(tmpdir, info_text=info)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=[[16, 16, 40]],
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(tmp_layer, on_info_exists="overwrite")


def test_add_scale_with_overrides_okay(clear_caches_reset_mocks, tmpdir):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=[[16, 16, 40]],
        field_overrides={"type": "uint8"},
    )
    info_spec.update_info(tmp_layer, on_info_exists="overwrite")


def test_add_scale_with_overrides_error(clear_caches_reset_mocks, tmpdir):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales=[[16, 16, 40]],
        field_overrides={
            "scales": [
                {
                    "chunk_sizes": [[1024, 1024, 1]],
                    "encoding": "raw",
                    "key": "4_4_40",
                    "resolution": [4, 4, 40],
                    "size": [16384, 16384, 16384],
                    "voxel_offset": [0, 0, 0],
                }
            ]
        },
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(tmp_layer, on_info_exists="overwrite")


def test_add_scale_with_custom_ref(clear_caches_reset_mocks, tmpdir):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales_ref={
            "key": "3_3_41",
            "resolution": [3, 3, 41],
            "chunk_sizes": [[512, 512, 1]],
            "encoding": "raw",
            "size": [1000, 1000, 1000],
            "voxel_offset": [2, 4, 6],
        },
        add_scales=[[6, 6, 82]],
    )
    info_spec.update_info(tmp_layer, on_info_exists="overwrite")
    scales = get_info(tmp_layer)["scales"]
    assert "6_6_82" in [e["key"] for e in scales]


def test_add_scale_with_non_existing_ref(clear_caches_reset_mocks, tmpdir):
    tmp_layer = make_tmp_layer(tmpdir)
    info_spec = PrecomputedInfoSpec(
        reference_path=tmp_layer,
        add_scales_ref="32_32_40",
        add_scales=[[16, 16, 40]],
    )
    with pytest.raises(RuntimeError):
        info_spec.update_info(tmp_layer, on_info_exists="overwrite")
