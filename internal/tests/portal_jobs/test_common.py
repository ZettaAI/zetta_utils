import pytest

from zetta_utils.internal.portal_jobs.common import get_chunk_size


@pytest.mark.parametrize(
    "chunk_size_base, bbox_size, must_be_divisible, expected",
    [
        [(1024, 1024, 1024), (512, 256, 128), True, (512, 256, 128)],
        [(1024, 1024, 1024), (512, 256, 128), False, (512, 256, 128)],
        [(500, 400, 300), (1024, 1024, 1024), False, (500, 400, 300)],
        [(500, 400, 300), (1024, 1024, 1024), True, (256, 256, 256)],
        [(8 * 1024, 8 * 1024, 8), (45831, 37326, 1), False, (8 * 1024, 8 * 1024, 1)],
    ],
)
def test_get_chunk_size(chunk_size_base, bbox_size, must_be_divisible, expected):
    result = get_chunk_size(
        chunk_size_base=chunk_size_base, bbox_size=bbox_size, must_be_divisible=must_be_divisible
    )
    assert result == expected
