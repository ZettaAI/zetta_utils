# pylint: disable=missing-docstring,unspecified-encoding,invalid-name
import os
import pathlib

from cloudfiles import CloudFiles

from zetta_utils.bcube import BoundingCube
from zetta_utils.parsing import ngl_state

THIS_DIR = pathlib.Path(__file__).parent.resolve()
TEST_DATA_DIR = THIS_DIR / "../assets/remote_layers/"
TEST_READ_FILE = "ngl_layer.json"
TEST_WRITE_FILE = "ngl_layer_write.json"
REMOTE_LAYERS_PATH = f"file://{TEST_DATA_DIR}"


def test_read_remote_annotations():
    os.environ["REMOTE_LAYERS_PATH"] = REMOTE_LAYERS_PATH
    result = ngl_state.read_remote_annotations(TEST_READ_FILE)
    assert len(result) == 2


def test_write_remote_annotations():
    start_coord = [52632.3828125, 73167.4140625, 1040]
    end_coord = [54340.7421875, 74262.7265625, 1041]
    layer_name = TEST_WRITE_FILE
    resolution = [4, 4, 30]

    bcubes_or_points = [
        BoundingCube.from_coords(start_coord, end_coord, resolution),
        [181038.53125, 283613.8125, 28860.00183105],
    ]

    ngl_state.write_remote_annotations(layer_name, resolution, bcubes_or_points)
    cf = CloudFiles(REMOTE_LAYERS_PATH)
    layer_json = cf.get_json(TEST_WRITE_FILE)
    assert len(layer_json["annotations"]) == 2
