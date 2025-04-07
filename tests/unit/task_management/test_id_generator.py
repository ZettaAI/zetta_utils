from zetta_utils.task_management.id_generator import (
    MAX_ID_NONUNIQUE,
    generate_nonunique_id,
)


def test_generate_nonunique_id():
    """Test that generate_nonunique_id returns an integer within the expected range."""
    # Generate multiple IDs to ensure they're within range
    for _ in range(100):
        id_value = generate_nonunique_id()
        assert isinstance(id_value, int)
        assert 0 <= id_value <= MAX_ID_NONUNIQUE


def test_max_id_nonunique():
    """Test that MAX_ID_NONUNIQUE is set to 2^64-1."""
    assert MAX_ID_NONUNIQUE == (2 ** 64) - 1
