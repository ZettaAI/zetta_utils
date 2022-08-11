import pathlib
import pickle

import zetta_utils as zu
from zetta_utils import training  # pylint: disable=unused-import

THIS_DIR = pathlib.Path(__file__).parent.resolve()
TEST_DATA_DIR = THIS_DIR / "../assets/"


def test_dataset_builder():
    dset_spec_path = TEST_DATA_DIR / "params/training_x0/datasets/dataset_x0.cue"
    result = zu.builder.build(zu.cue.load(dset_spec_path))
    reference_path = TEST_DATA_DIR / "reference/built_dataset_x0_sample_100_data_in.pkl"
    data = result[100]["data_in"]
    # with open(reference_path, "wb") as f:
    # pickle.dump(data, f)

    with open(reference_path, "rb") as f:
        reference = pickle.load(f)

    assert (data == reference).all()
