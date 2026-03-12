import os
from concurrent.futures import Future
from typing import Any, Dict

import lightning.pytorch as pl
import torch
import torch.multiprocessing as mp

from zetta_utils import training
from zetta_utils.training.lightning.trainers.default import (
    ConfigureLogging,
    _apply_map_location,
    _fix_dcp_optimizer_keys,
    _is_dcp_checkpoint,
    _load_dcp_checkpoint,
    _save_checkpoint_worker,
    _stage_checkpoint,
    _upload_checkpoint,
    _wait_and_upload,
)


def test_default_trainer():
    result = training.lightning.trainers.ZettaDefaultTrainer(
        experiment_name="unit_test",
        experiment_version="x0",
    )
    assert isinstance(result, pl.Trainer)


# --- _stage_checkpoint ---


def test_stage_checkpoint_tensor():
    t = torch.tensor([1.0, 2.0, 3.0])
    staged = _stage_checkpoint(t)
    assert staged.is_shared()
    assert torch.equal(staged, t)
    staged[0] = 99.0
    assert t[0] == 1.0


def test_stage_checkpoint_dict():
    data: Dict[str, Any] = {"a": torch.tensor([1.0]), "b": 42}
    staged = _stage_checkpoint(data)
    assert isinstance(staged, dict)
    assert staged["a"].is_shared()
    assert torch.equal(staged["a"], torch.tensor([1.0]))
    assert staged["b"] == 42


def test_stage_checkpoint_list_tuple():
    data_list = [torch.tensor([1.0]), "hello"]
    staged_list = _stage_checkpoint(data_list)
    assert isinstance(staged_list, list)
    assert staged_list[0].is_shared()
    assert staged_list[1] == "hello"

    data_tuple = (torch.tensor([2.0]), 3)
    staged_tuple = _stage_checkpoint(data_tuple)
    assert isinstance(staged_tuple, tuple)
    assert staged_tuple[0].is_shared()
    assert staged_tuple[1] == 3


def test_stage_checkpoint_primitive():
    assert _stage_checkpoint(42) == 42
    assert _stage_checkpoint("hello") == "hello"
    assert _stage_checkpoint(None) is None


# --- _is_dcp_checkpoint ---


def test_is_dcp_checkpoint_true(tmp_path):
    dcp_dir = tmp_path / "ckpt"
    dcp_dir.mkdir()
    (dcp_dir / ".metadata").touch()
    assert _is_dcp_checkpoint(str(dcp_dir)) is True


def test_is_dcp_checkpoint_no_metadata(tmp_path):
    dcp_dir = tmp_path / "ckpt"
    dcp_dir.mkdir()
    assert _is_dcp_checkpoint(str(dcp_dir)) is False


def test_is_dcp_checkpoint_file(tmp_path):
    ckpt_file = tmp_path / "ckpt.pt"
    ckpt_file.touch()
    assert _is_dcp_checkpoint(str(ckpt_file)) is False


def test_is_dcp_checkpoint_nonexistent(tmp_path):
    assert _is_dcp_checkpoint(str(tmp_path / "nonexistent")) is False


def test_is_dcp_checkpoint_remote_true(tmp_path):
    dcp_dir = tmp_path / "ckpt"
    dcp_dir.mkdir()
    (dcp_dir / ".metadata").touch()
    assert _is_dcp_checkpoint(f"file://{dcp_dir}") is True


def test_is_dcp_checkpoint_remote_false(tmp_path):
    dcp_dir = tmp_path / "ckpt"
    dcp_dir.mkdir()
    assert _is_dcp_checkpoint(f"file://{dcp_dir}") is False


# --- _fix_dcp_optimizer_keys ---


def test_fix_dcp_optimizer_keys_converts_string_digits():
    checkpoint: Dict[str, Any] = {
        "optimizer_states": [
            {
                "state": {
                    "0": {"momentum": torch.tensor([1.0])},
                    "1": {"momentum": torch.tensor([2.0])},
                }
            }
        ]
    }
    _fix_dcp_optimizer_keys(checkpoint)
    assert set(checkpoint["optimizer_states"][0]["state"].keys()) == {0, 1}


def test_fix_dcp_optimizer_keys_leaves_non_digit():
    checkpoint: Dict[str, Any] = {
        "optimizer_states": [
            {
                "state": {
                    "param_name": {"lr": 0.01},
                    "0": {"momentum": torch.tensor([1.0])},
                }
            }
        ]
    }
    _fix_dcp_optimizer_keys(checkpoint)
    keys = set(checkpoint["optimizer_states"][0]["state"].keys())
    assert "param_name" in keys
    assert 0 in keys


def test_fix_dcp_optimizer_keys_empty_list():
    checkpoint: Dict[str, Any] = {"optimizer_states": []}
    _fix_dcp_optimizer_keys(checkpoint)
    assert not checkpoint["optimizer_states"]


def test_fix_dcp_optimizer_keys_missing_key():
    checkpoint: Dict[str, Any] = {"epoch": 5}
    _fix_dcp_optimizer_keys(checkpoint)
    assert checkpoint == {"epoch": 5}


def test_fix_dcp_optimizer_keys_non_dict_entry():
    checkpoint: Dict[str, Any] = {"optimizer_states": ["not_a_dict", 42]}
    _fix_dcp_optimizer_keys(checkpoint)
    assert checkpoint["optimizer_states"] == ["not_a_dict", 42]


# --- _apply_map_location ---


def test_apply_map_location_tensor():
    t = torch.tensor([1.0])
    result = _apply_map_location(t, "cpu")
    assert result.device == torch.device("cpu")
    assert torch.equal(result, t)


def test_apply_map_location_nested_dict():
    inner = {"b": torch.tensor([1.0])}
    data: Dict[str, Any] = {"a": inner, "c": 42}
    result = _apply_map_location(data, "cpu")
    assert torch.equal(result["a"]["b"], inner["b"])
    assert result["c"] == 42


def test_apply_map_location_list_tuple():
    t0 = torch.tensor([1.0])
    t1 = torch.tensor([2.0])
    data = [t0, (t1, 3)]
    result = _apply_map_location(data, "cpu")
    assert isinstance(result, list)
    assert isinstance(result[1], tuple)
    assert torch.equal(result[0], t0)
    assert result[1][1] == 3


def test_apply_map_location_primitive():
    assert _apply_map_location(42, "cpu") == 42
    assert _apply_map_location("hello", "cpu") == "hello"


# --- _upload_checkpoint ---


def test_upload_checkpoint_local(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "file1.txt").write_text("data1")
    (src / "file2.txt").write_text("data2")

    dest = tmp_path / "dest"
    _upload_checkpoint(str(src), str(dest))

    assert (dest / "file1.txt").read_text() == "data1"
    assert (dest / "file2.txt").read_text() == "data2"


def test_upload_checkpoint_overwrites(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "new.txt").write_text("new_data")

    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "old.txt").write_text("old_data")

    _upload_checkpoint(str(src), str(dest))

    assert (dest / "new.txt").read_text() == "new_data"
    assert not (dest / "old.txt").exists()


def test_upload_checkpoint_remote(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "file1.txt").write_text("data1")
    (src / "file2.txt").write_text("data2")

    dest = tmp_path / "dest"
    dest.mkdir()
    _upload_checkpoint(str(src), f"file://{dest}")

    assert (dest / "file1.txt").read_text() == "data1"
    assert (dest / "file2.txt").read_text() == "data2"


# --- _save_checkpoint_worker ---


def test_save_checkpoint_worker_success(tmp_path):
    checkpoint = {"epoch": 1, "state": torch.tensor([1.0, 2.0])}
    path = str(tmp_path / "ckpt.pt")
    queue: mp.Queue = mp.Queue()

    _save_checkpoint_worker(checkpoint, path, None, queue)

    assert queue.empty()
    loaded = torch.load(path, weights_only=False)
    assert loaded["epoch"] == 1


def test_save_checkpoint_worker_failure(tmp_path):
    queue: mp.Queue = mp.Queue()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    bad_path = str(blocker / "sub" / "ckpt.pt")

    _save_checkpoint_worker({"x": 1}, bad_path, None, queue)

    err = queue.get(timeout=5)
    assert "Error" in err


# --- _wait_and_upload ---


def test_wait_and_upload_success(tmp_path):
    src = tmp_path / "local"
    src.mkdir()
    (src / "data.bin").write_text("checkpoint_data")

    dest = tmp_path / "remote"
    error_holder: list[BaseException] = []

    future: Future[None] = Future()
    future.set_result(None)

    _wait_and_upload(future, str(src), str(dest), error_holder)

    assert len(error_holder) == 0
    assert (dest / "data.bin").read_text() == "checkpoint_data"
    assert not src.exists()


def test_wait_and_upload_failure():
    error_holder: list[BaseException] = []

    future: Future[None] = Future()
    future.set_exception(RuntimeError("save failed"))

    _wait_and_upload(future, "/nonexistent", "/also_nonexistent", error_holder)

    assert len(error_holder) == 1
    assert isinstance(error_holder[0], RuntimeError)


# --- _load_dcp_checkpoint ---


def test_load_dcp_checkpoint_local(tmp_path, mocker):
    load_path = str(tmp_path / "dcp_ckpt")
    os.makedirs(load_path)

    fake_checkpoint: Dict[str, Any] = {"epoch": 5, "global_step": 100}

    def mock_load_state_dict(state_dict, **_kwargs):
        state_dict.update(fake_checkpoint)

    mocker.patch(
        "zetta_utils.training.lightning.trainers.default._load_state_dict",
        side_effect=mock_load_state_dict,
    )

    result = _load_dcp_checkpoint(load_path)

    assert result["epoch"] == 5
    assert result["global_step"] == 100


def test_load_dcp_checkpoint_with_map_location(tmp_path, mocker):
    load_path = str(tmp_path / "dcp_ckpt")
    os.makedirs(load_path)

    fake_checkpoint: Dict[str, Any] = {"state": torch.tensor([1.0])}

    def mock_load_state_dict(state_dict, **_kwargs):
        state_dict.update(fake_checkpoint)

    mocker.patch(
        "zetta_utils.training.lightning.trainers.default._load_state_dict",
        side_effect=mock_load_state_dict,
    )

    result = _load_dcp_checkpoint(load_path, map_location="cpu")
    assert result["state"].device == torch.device("cpu")


def test_load_dcp_checkpoint_fixes_optimizer_keys(tmp_path, mocker):
    load_path = str(tmp_path / "dcp_ckpt")
    os.makedirs(load_path)

    fake_checkpoint: Dict[str, Any] = {
        "optimizer_states": [{"state": {"0": {"momentum": torch.tensor([1.0])}}}]
    }

    def mock_load_state_dict(state_dict, **_kwargs):
        state_dict.update(fake_checkpoint)

    mocker.patch(
        "zetta_utils.training.lightning.trainers.default._load_state_dict",
        side_effect=mock_load_state_dict,
    )

    result = _load_dcp_checkpoint(load_path)
    assert 0 in result["optimizer_states"][0]["state"]


def test_load_dcp_checkpoint_remote(tmp_path, mocker):
    dcp_dir = tmp_path / "dcp_ckpt"
    dcp_dir.mkdir()
    (dcp_dir / ".metadata").write_text("fake")
    (dcp_dir / "__0_0.distcp").write_text("fake")

    fake_checkpoint: Dict[str, Any] = {"epoch": 10}

    def mock_load_state_dict(state_dict, **_kwargs):
        state_dict.update(fake_checkpoint)

    mocker.patch(
        "zetta_utils.training.lightning.trainers.default._load_state_dict",
        side_effect=mock_load_state_dict,
    )

    result = _load_dcp_checkpoint(f"file://{dcp_dir}")
    assert result["epoch"] == 10


# --- ConfigureLogging ---


def test_configure_logging_init():
    cb = ConfigureLogging("my_exp", "v1")
    assert cb.exp_name == "my_exp"
    assert cb.exp_version == "v1"
