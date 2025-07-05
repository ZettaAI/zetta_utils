# pylint: disable=redefined-outer-name,unused-argument

from zetta_utils.task_management.utils import MAX_NONUNIQUE_ID, generate_id_nonunique


def test_generate_id_nonunique():
    """Test that generate_id_nonunique returns a valid ID within range"""
    id_value = generate_id_nonunique()
    assert isinstance(id_value, int)
    assert 0 <= id_value <= MAX_NONUNIQUE_ID


def test_generate_id_nonunique_multiple_calls():
    """Test that multiple calls can generate different IDs (though not guaranteed)"""
    ids = [generate_id_nonunique() for _ in range(10)]
    # All should be valid
    for id_value in ids:
        assert isinstance(id_value, int)
        assert 0 <= id_value <= MAX_NONUNIQUE_ID

    # At least check they're all integers in valid range
    assert len(ids) == 10
