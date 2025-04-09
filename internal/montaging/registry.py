# pylint: disable = invalid-name
from __future__ import annotations

from itertools import islice
from typing import Any, Generator

from zetta_utils.layer.db_layer import DBLayer


def zs_to_tiles(
    tile_registry: DBLayer,
    z_start: int,
    z_stop: int,
) -> dict[str, dict[str, Any]]:
    """
    Returns all tiles with Z offset between z_start and z_stop.

    :param tile_registry: The tile registry.
    :param z_start: Z offset to start, inclusive.
    :param z_stop: Z offset to stop, exclusive.

    """
    tiles_in_zs = tile_registry.query({"z_offset": list(range(z_start, z_stop))})
    data = tile_registry[
        tiles_in_zs,
        (
            "x_offset",
            "y_offset",
            "z_offset",
            "x_index",
            "y_index",
            "z_index",
            "x_size",
            "y_size",
            "x_res",
            "y_res",
            "z_res",
        ),
    ]
    return dict(zip(tiles_in_zs, data))


# TODO: When upgrading to Python 3.12, use itertools.batched
def dict_to_chunks(data: dict, size: int) -> Generator[dict, None, None]:
    """
    Splits a dictionary into dictionaries of size at most ``size``.

    :param data: Initial dictionary.
    :param size: Max size of the returned dictionaries.

    """
    it = iter(data)
    for _ in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


"""
This function is only here as an example of a registry import function.
The `stage_positions.csv` file referenced is from the customer, and has as columns:
tile_id, stage_x_nm, stage_y_nm, x_relroi_nm, y_relroi_nm.
The stage_x_nm and stage_y_nm are not directly used except to compute the XY indices.
x_res, y_res, x_size, y_size were given by the customer, and exp_offset_nominal
was empirically found.

You should use batched writes as in this example rather than a for loop for writing
to a remote registry; it's extremely slow to do otherwise.
"""
"""
def z_to_tiles_to_registry(
    z: int,
    registry: DBLayer,
    bucket="gs://ng_scratch_ranl_7/test_voxa/tiles/",
    stage_positions_csv="~/stage_positions.csv",
):

    import pandas
    x_res = 3.382
    y_res = 3.646
    x_size = 5496
    y_size = 5496
    exp_offset_nominal = 4672
    stage_positions = pandas.read_csv(stage_positions_csv, header=None, sep=",").to_numpy()[1:]
    z_to_prefix = {
        0: "2022.10.01_Sample3_tilt+0/s000.01-2022.10.02-01.04.56",
        -1: "2022.10.01_Sample3_tilt-5/s000.01-2022.10.02-20.07.42",
        +1: "2022.10.01_Sample3_tilt+5/s000.01-2022.10.02-02.02.36",
        -2: "2022.10.01_Sample3_tilt-10/s000.01-2022.10.02-19.35.14",
        +2: "2022.10.01_Sample3_tilt+10/s000.01-2022.10.02-02.37.12",
        -3: "2022.10.09_Sample3_tilt-15b/s000.01-2022.10.09-19.41.25",
        +3: "2022.10.09_Sample3_tilt+15b/s000.01-2022.10.09-20.58.34",
        -4: "2022.10.07_Sample3_tilt-20b/s000.01-2022.10.07-17.27.40",
        +4: "2022.10.07_Sample3_tilt+20b/s000.01-2022.10.07-18.34.20",
        -5: "2022.10.07_Sample3_tilt-25b/s000.01-2022.10.07-16.03.37",
        +5: "2022.10.07_Sample3_tilt+25b/s000.01-2022.10.07-19.30.44",
    }
    tile_names = [
        os.path.join(bucket, z_to_prefix[z], f"tile_{str(row[0]).zfill(4)}.bmp")
        for row in stage_positions
    ]
    xs = [int(round(float(row[3]) / x_res)) for row in stage_positions]
    # ONLY FOR HIVE_TOMOGRAPHY
    ys = [-1 * int(round(float(row[4]) / y_res)) for row in stage_positions]
    zs = [z for row in stage_positions]
    x_inds = [int(round(float(row[3]) / x_res / exp_offset_nominal)) for row in stage_positions]
    # ONLY FOR HIVE_TOMOGRAPHY
    y_inds = [
        -1 * int(round(float(row[4]) / y_res / exp_offset_nominal)) for row in stage_positions
    ]
    z_inds = [z for row in stage_positions]
    tiles = list(zip(tile_names, zip(xs, ys, zs, x_inds, y_inds, z_inds)))
    tiles_filtered = dict(
        filter(lambda tile: -10 < tile[1][3] < 10 and -10 < tile[1][4] < 10, tiles)
    )

    x_res = 4
    y_res = 4
    z_res = 1

    tiles_list = list(tiles_filtered.keys())
    datas = [
        {
            "x_offset": data[0],
            "y_offset": data[1],
            "z_offset": data[2],
            "x_index": data[3],
            "y_index": data[4],
            "z_index": data[5],
            "x_size": x_size,
            "y_size": y_size,
            "x_res": x_res,
            "y_res": y_res,
            "z_res": z_res,
        }
        for data in tiles_filtered.values()
    ]
    registry[
        tiles_list,
        (
            "x_offset",
            "y_offset",
            "z_offset",
            "x_index",
            "y_index",
            "z_index",
            "x_size",
            "y_size",
            "x_res",
            "y_res",
            "z_res",
        ),
    ] = datas
"""
