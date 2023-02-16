import fsspec

from zetta_utils import builder


@builder.register("test_gcs_access")
def test_gcs_access(read_path: str, write_path):
    with fsspec.open(read_path, "r") as f:
        data_read_x0 = f.read()

    with fsspec.open(write_path, "w") as f:
        f.write(data_read_x0)

    with fsspec.open(write_path, "r") as f:
        data_read_x1 = f.read()

    assert data_read_x1 == data_read_x0
