# pylint: disable=redefined-outer-name
import os
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import lightning.pytorch as pl
import pytest
import torch

from zetta_utils import training
from zetta_utils.training.lightning.trainers.default import (
    ZettaDefaultTrainer,
    jit_trace_export,
    onnx_export,
)


def test_default_trainer():
    result = training.lightning.trainers.ZettaDefaultTrainer(
        experiment_name="unit_test",
        experiment_version="x0",
    )
    assert isinstance(result, pl.Trainer)


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_trace_input():
    return torch.randn(1, 10)


@pytest.fixture
def mock_lightning_module():
    """Fixture providing a mock lightning module."""
    mock_module = MagicMock()
    mock_module._modules = {}  # pylint: disable=protected-access
    return mock_module


@pytest.fixture
def trainer_mocks(mocker):
    """Fixture providing common mocks for trainer tests."""
    return {
        "super_save": mocker.patch.object(pl.Trainer, "save_checkpoint"),
        "jit_export": mocker.patch(
            "zetta_utils.training.lightning.trainers.default.jit_trace_export"
        ),
        "onnx_export": mocker.patch("zetta_utils.training.lightning.trainers.default.onnx_export"),
        "is_global_zero": mocker.patch.object(
            ZettaDefaultTrainer, "is_global_zero", new_callable=PropertyMock
        ),
        "lightning_module": mocker.patch.object(
            ZettaDefaultTrainer, "lightning_module", new_callable=PropertyMock
        ),
    }


def test_save_checkpoint_calls_exports_when_enabled(
    trainer_mocks, mock_model, mock_trace_input, mock_lightning_module
):
    """Test that save_checkpoint calls export functions when exports are enabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ZettaDefaultTrainer(
            experiment_name="unit_test",
            experiment_version="x0",
            enable_jit_export=True,
            enable_onnx_export=True,
            default_root_dir=tmp_dir,
        )

        trainer.trace_configuration = {
            "test_model": {"model": mock_model, "trace_input": (mock_trace_input,)}
        }

        filepath = os.path.join(tmp_dir, "test_checkpoint.ckpt")

        trainer_mocks["is_global_zero"].return_value = True
        trainer_mocks["lightning_module"].return_value = mock_lightning_module

        trainer.save_checkpoint(filepath)

        trainer_mocks["onnx_export"].assert_called_once_with(
            mock_model, (mock_trace_input,), filepath, "test_model"
        )
        trainer_mocks["jit_export"].assert_called_once_with(
            mock_model, (mock_trace_input,), filepath, "test_model"
        )
        trainer_mocks["super_save"].assert_called_once()


def test_save_checkpoint_skips_jit_when_disabled(
    trainer_mocks, mock_model, mock_trace_input, mock_lightning_module
):
    """Test that save_checkpoint skips JIT export when disabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ZettaDefaultTrainer(
            experiment_name="unit_test",
            experiment_version="x0",
            enable_jit_export=False,
            enable_onnx_export=True,
            default_root_dir=tmp_dir,
        )

        trainer.trace_configuration = {
            "test_model": {"model": mock_model, "trace_input": (mock_trace_input,)}
        }

        filepath = os.path.join(tmp_dir, "test_checkpoint.ckpt")

        trainer_mocks["is_global_zero"].return_value = True
        trainer_mocks["lightning_module"].return_value = mock_lightning_module

        trainer.save_checkpoint(filepath)

        trainer_mocks["onnx_export"].assert_called_once_with(
            mock_model, (mock_trace_input,), filepath, "test_model"
        )
        trainer_mocks["jit_export"].assert_not_called()


def test_save_checkpoint_skips_onnx_when_disabled(
    trainer_mocks, mock_model, mock_trace_input, mock_lightning_module
):
    """Test that save_checkpoint skips ONNX export when disabled."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ZettaDefaultTrainer(
            experiment_name="unit_test",
            experiment_version="x0",
            enable_jit_export=True,
            enable_onnx_export=False,
            default_root_dir=tmp_dir,
        )

        trainer.trace_configuration = {
            "test_model": {"model": mock_model, "trace_input": (mock_trace_input,)}
        }

        filepath = os.path.join(tmp_dir, "test_checkpoint.ckpt")

        trainer_mocks["is_global_zero"].return_value = True
        trainer_mocks["lightning_module"].return_value = mock_lightning_module

        trainer.save_checkpoint(filepath)

        trainer_mocks["jit_export"].assert_called_once_with(
            mock_model, (mock_trace_input,), filepath, "test_model"
        )
        trainer_mocks["onnx_export"].assert_not_called()


def test_save_checkpoint_skips_exports_non_global_zero(trainer_mocks):
    """Test that save_checkpoint skips exports when not global zero rank."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ZettaDefaultTrainer(
            experiment_name="unit_test",
            experiment_version="x0",
            enable_jit_export=True,
            enable_onnx_export=True,
            default_root_dir=tmp_dir,
        )

        filepath = os.path.join(tmp_dir, "test_checkpoint.ckpt")

        trainer_mocks["is_global_zero"].return_value = False

        trainer.save_checkpoint(filepath)

        trainer_mocks["jit_export"].assert_not_called()
        trainer_mocks["onnx_export"].assert_not_called()
        trainer_mocks["super_save"].assert_called_once()


@patch("zetta_utils.training.lightning.trainers.default.logger")
def test_jit_trace_export_failure(mock_logger, mock_model, mock_trace_input):
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = os.path.join(tmp_dir, "test_model")

        with patch("torch.multiprocessing.get_context") as mock_ctx:
            mock_ctx.side_effect = RuntimeError("Mock export failure")

            jit_trace_export(mock_model, mock_trace_input, filepath, "test_model")

            mock_logger.warning.assert_called_once()
            assert "JIT trace export failed" in mock_logger.warning.call_args[0][0]


@patch("zetta_utils.training.lightning.trainers.default.logger")
def test_onnx_export_failure(mock_logger, mock_model, mock_trace_input):
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = os.path.join(tmp_dir, "test_model")

        with patch("torch.onnx.export") as mock_torch_onnx:
            mock_torch_onnx.side_effect = RuntimeError("Mock ONNX export failure")

            with patch("fsspec.open", create=True):
                onnx_export(mock_model, mock_trace_input, filepath, "test_model")

            mock_logger.warning.assert_called_once()
            assert "ONNX export failed" in mock_logger.warning.call_args[0][0]


def test_export_functions_preserve_training_mode(mock_model, mock_trace_input):
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = os.path.join(tmp_dir, "test_model")

        # Test with training mode enabled
        mock_model.train()
        original_mode = mock_model.training
        assert original_mode is True

        # Test ONNX export preserves training mode
        with patch("torch.onnx.export"):
            with patch("fsspec.open", create=True):
                onnx_export(mock_model, mock_trace_input, filepath, "test_model")

        assert mock_model.training == original_mode

        # Test JIT export preserves training mode
        with patch("torch.multiprocessing.get_context") as mock_ctx:
            mock_process = MagicMock()
            mock_ctx.return_value.Process.return_value = mock_process

            jit_trace_export(mock_model, mock_trace_input, filepath, "test_model")

        assert mock_model.training == original_mode


def test_multiple_models_export(trainer_mocks, mock_trace_input, mock_lightning_module):
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ZettaDefaultTrainer(
            experiment_name="unit_test",
            experiment_version="x0",
            enable_jit_export=True,
            enable_onnx_export=True,
            default_root_dir=tmp_dir,
        )

        model1 = MockModel()
        model2 = MockModel()

        trainer.trace_configuration = {
            "model1": {"model": model1, "trace_input": (mock_trace_input,)},
            "model2": {"model": model2, "trace_input": (mock_trace_input,)},
        }

        filepath = os.path.join(tmp_dir, "test_checkpoint.ckpt")

        trainer_mocks["is_global_zero"].return_value = True
        trainer_mocks["lightning_module"].return_value = mock_lightning_module

        trainer.save_checkpoint(filepath)

        # Verify both models were exported
        assert trainer_mocks["jit_export"].call_count == 2
        assert trainer_mocks["onnx_export"].call_count == 2

        # Check that both models were called with correct names
        jit_calls = [call[0] for call in trainer_mocks["jit_export"].call_args_list]
        onnx_calls = [call[0] for call in trainer_mocks["onnx_export"].call_args_list]

        assert any(call[0] is model1 and call[3] == "model1" for call in jit_calls)
        assert any(call[0] is model2 and call[3] == "model2" for call in jit_calls)
        assert any(call[0] is model1 and call[3] == "model1" for call in onnx_calls)
        assert any(call[0] is model2 and call[3] == "model2" for call in onnx_calls)
