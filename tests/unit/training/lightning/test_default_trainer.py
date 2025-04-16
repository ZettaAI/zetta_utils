import lightning.pytorch as pl

from zetta_utils import training


def test_default_trainer():
    result = training.lightning.trainers.ZettaDefaultTrainer(
        experiment_name="unit_test",
        experiment_version="x0",
    )
    assert isinstance(result, pl.Trainer)
