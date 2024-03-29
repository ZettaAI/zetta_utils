# pylint: disable=missing-docstring,unspecified-encoding,invalid-name
import os
import pathlib
from typing import Any, List, Union

from cloudfiles import CloudFiles

from zetta_utils.geometry import BBox3D, Vec3D
from zetta_utils.parsing import ngl_state

PARENT_DIR = pathlib.Path(__file__).parents[1]
TEST_DATA_DIR = "assets/remote_layers/"
TEST_READ_FILE = "ngl_layer.json"
TEST_WRITE_FILE = "ngl_layer_write.json"
REMOTE_LAYERS_PATH = f"file://{PARENT_DIR}/{TEST_DATA_DIR}"
os.environ["REMOTE_LAYERS_PATH"] = REMOTE_LAYERS_PATH


def test_read_remote_annotations():
    result = ngl_state.read_remote_annotations(TEST_READ_FILE)
    assert len(result) == 2


def test_write_remote_annotations():
    start_coord = Vec3D(52632.3828125, 73167.4140625, 1040)
    end_coord = Vec3D(54340.7421875, 74262.7265625, 1041)
    resolution = Vec3D(4, 4, 30)
    layer_name = TEST_WRITE_FILE

    bboxes_or_points: List[Union[BBox3D, Vec3D[Any]]] = [
        BBox3D.from_coords(start_coord, end_coord, resolution),
        Vec3D(181038.53125, 283613.8125, 28860.00183105),
    ]

    ngl_state.write_remote_annotations(layer_name, resolution, bboxes_or_points)
    cf = CloudFiles(REMOTE_LAYERS_PATH)
    layer_json = cf.get_json(TEST_WRITE_FILE)
    assert len(layer_json["annotations"]) == 2
