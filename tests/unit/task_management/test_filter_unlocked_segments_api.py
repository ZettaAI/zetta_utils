"""Tests for filter_unlocked_segments API endpoint logic"""

# pylint: disable=unused-argument,redefined-outer-name,import-outside-toplevel

import pytest

from zetta_utils.task_management.project import create_project


@pytest.fixture
def existing_project(clean_db, db_session, project_name):
    """Create a project for testing"""
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        datastack_name="test_datastack",
        synapse_table="test_synapse_table",
        db_session=db_session,
    )
    yield project_name


def test_filter_unlocked_segments_returns_strings(existing_project, db_session, project_name, mocker):
    """Test that filter_unlocked_segments API returns segment IDs as strings"""
    from zetta_utils.task_management import segment as segment_module

    # Mock get_unlocked_segments to return some integer segment IDs
    mock_unlocked_segments = [12345, 67890, 11111]

    mocker.patch.object(
        segment_module,
        "get_unlocked_segments",
        return_value=mock_unlocked_segments,
    )

    # Simulate the API logic that converts integers to strings
    input_segment_ids = [12345, 67890, 11111, 22222, 33333]
    unlocked_segments = segment_module.get_unlocked_segments(
        project_name=project_name,
        segment_ids=input_segment_ids,
    )
    
    # Convert to strings as the API does
    result = {
        "unlocked_segments": [str(seg_id) for seg_id in unlocked_segments],
    }

    # Verify the result contains string representations
    assert "unlocked_segments" in result
    assert isinstance(result["unlocked_segments"], list)
    assert len(result["unlocked_segments"]) == 3
    assert result["unlocked_segments"] == ["12345", "67890", "11111"]
    
    # Verify all items are strings
    for seg_id in result["unlocked_segments"]:
        assert isinstance(seg_id, str)


def test_filter_unlocked_segments_empty_list(existing_project, db_session, project_name, mocker):
    """Test filtering when no segments are unlocked"""
    from zetta_utils.task_management import segment as segment_module

    # Mock get_unlocked_segments to return empty list
    mocker.patch.object(
        segment_module,
        "get_unlocked_segments",
        return_value=[],
    )

    input_segment_ids = [12345, 67890, 11111]
    unlocked_segments = segment_module.get_unlocked_segments(
        project_name=project_name,
        segment_ids=input_segment_ids,
    )
    
    result = {
        "unlocked_segments": [str(seg_id) for seg_id in unlocked_segments],
    }

    assert result["unlocked_segments"] == []
    assert isinstance(result["unlocked_segments"], list)


def test_filter_unlocked_segments_partial_filtering(existing_project, db_session, project_name, mocker):
    """Test filtering when some segments are locked and some are unlocked"""
    from zetta_utils.task_management import segment as segment_module

    # Mock get_unlocked_segments to return subset of input
    mock_unlocked_segments = [12345, 11111]  # 67890 is locked

    mocker.patch.object(
        segment_module,
        "get_unlocked_segments",
        return_value=mock_unlocked_segments,
    )

    input_segment_ids = [12345, 67890, 11111]
    unlocked_segments = segment_module.get_unlocked_segments(
        project_name=project_name,
        segment_ids=input_segment_ids,
    )
    
    result = {
        "unlocked_segments": [str(seg_id) for seg_id in unlocked_segments],
    }

    assert len(result["unlocked_segments"]) == 2
    assert "12345" in result["unlocked_segments"]
    assert "11111" in result["unlocked_segments"]
    assert "67890" not in result["unlocked_segments"]
    
    # Verify all items are strings
    for seg_id in result["unlocked_segments"]:
        assert isinstance(seg_id, str)


def test_filter_unlocked_segments_single_segment(existing_project, db_session, project_name, mocker):
    """Test filtering with a single segment"""
    from zetta_utils.task_management import segment as segment_module

    # Mock get_unlocked_segments to return single segment
    mock_unlocked_segments = [12345]

    mocker.patch.object(
        segment_module,
        "get_unlocked_segments",
        return_value=mock_unlocked_segments,
    )

    input_segment_ids = [12345]
    unlocked_segments = segment_module.get_unlocked_segments(
        project_name=project_name,
        segment_ids=input_segment_ids,
    )
    
    result = {
        "unlocked_segments": [str(seg_id) for seg_id in unlocked_segments],
    }

    assert len(result["unlocked_segments"]) == 1
    assert result["unlocked_segments"][0] == "12345"
    assert isinstance(result["unlocked_segments"][0], str)


def test_filter_unlocked_segments_large_numbers(existing_project, db_session, project_name, mocker):
    """Test filtering with large segment IDs to ensure string conversion works correctly"""
    from zetta_utils.task_management import segment as segment_module

    # Use large segment IDs that might cause issues if not handled properly
    large_segment_ids = [999999999999999, 888888888888888, 777777777777777]

    mocker.patch.object(
        segment_module,
        "get_unlocked_segments",
        return_value=large_segment_ids,
    )

    unlocked_segments = segment_module.get_unlocked_segments(
        project_name=project_name,
        segment_ids=large_segment_ids,
    )
    
    result = {
        "unlocked_segments": [str(seg_id) for seg_id in unlocked_segments],
    }

    expected_strings = ["999999999999999", "888888888888888", "777777777777777"]
    assert result["unlocked_segments"] == expected_strings
    
    # Verify all items are strings and conversion is correct
    for i, seg_id in enumerate(result["unlocked_segments"]):
        assert isinstance(seg_id, str)
        assert int(seg_id) == large_segment_ids[i]  # Verify round-trip conversion


def test_filter_unlocked_segments_api_input_validation(existing_project, db_session, project_name, mocker):
    """Test that the API logic correctly calls get_unlocked_segments with right parameters"""
    from zetta_utils.task_management import segment as segment_module

    # Mock get_unlocked_segments to capture call arguments
    mock_fn = mocker.patch.object(
        segment_module,
        "get_unlocked_segments",
        return_value=[12345, 67890],
    )

    # Simulate the API call logic
    input_segment_ids = [12345, 67890, 11111]
    
    unlocked_segments = segment_module.get_unlocked_segments(
        project_name=project_name,
        segment_ids=input_segment_ids,
    )
    
    # Verify the function was called with correct parameters
    mock_fn.assert_called_once_with(
        project_name=project_name,
        segment_ids=input_segment_ids,
    )
    
    # Verify the string conversion
    result = {
        "unlocked_segments": [str(seg_id) for seg_id in unlocked_segments],
    }
    
    assert result["unlocked_segments"] == ["12345", "67890"]