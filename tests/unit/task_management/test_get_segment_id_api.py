"""Tests for get_segment_id API endpoint logic"""

# pylint: disable=unused-argument,redefined-outer-name

import pytest

from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.segment import get_segment_id


@pytest.fixture
def existing_project(clean_db, db_session, project_name):
    """Create a project with segmentation info"""
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


def test_get_segment_id_success(existing_project, db_session, project_name, mocker):
    """Test getting segment IDs for multiple coordinates"""
    # Mock get_segment_id to return predictable values
    mock_segment_ids = {
        (100.0, 200.0, 300.0): 12345,
        (150.0, 250.0, 350.0): 67890,
        (200.0, 300.0, 400.0): 11111,
    }

    def mock_get_segment_id(project_name, coordinate, initial, db_session):
        coord_tuple = tuple(coordinate)
        return mock_segment_ids.get(coord_tuple, 0)

    mocker.patch(
        "zetta_utils.task_management.segment.get_segment_id",
        side_effect=mock_get_segment_id,
    )

    # Test the logic directly
    coordinates = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0], [200.0, 300.0, 400.0]]
    result = {}

    for coordinate in coordinates:
        segment_id = get_segment_id(
            project_name=project_name,
            coordinate=coordinate,
            initial=False,
            db_session=db_session,
        )
        if segment_id != 0:
            result[segment_id] = coordinate

    assert len(result) == 3
    assert result[12345] == [100.0, 200.0, 300.0]
    assert result[67890] == [150.0, 250.0, 350.0]
    assert result[11111] == [200.0, 300.0, 400.0]


def test_get_segment_id_filters_zero_segments(existing_project, db_session, project_name, mocker):
    """Test that segment_id=0 is filtered out"""
    # Mock get_segment_id to return some zeros
    mock_segment_ids = {
        (100.0, 200.0, 300.0): 12345,
        (150.0, 250.0, 350.0): 0,  # Should be filtered out
        (200.0, 300.0, 400.0): 67890,
    }

    def mock_get_segment_id(project_name, coordinate, initial, db_session):
        coord_tuple = tuple(coordinate)
        return mock_segment_ids.get(coord_tuple, 0)

    mocker.patch(
        "zetta_utils.task_management.segment.get_segment_id",
        side_effect=mock_get_segment_id,
    )

    coordinates = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0], [200.0, 300.0, 400.0]]
    result = {}

    for coordinate in coordinates:
        segment_id = get_segment_id(
            project_name=project_name,
            coordinate=coordinate,
            initial=False,
            db_session=db_session,
        )
        if segment_id != 0:
            result[segment_id] = coordinate

    assert len(result) == 2
    assert result[12345] == [100.0, 200.0, 300.0]
    assert result[67890] == [200.0, 300.0, 400.0]
    assert 0 not in result


def test_get_segment_id_with_initial_flag(existing_project, db_session, project_name, mocker):
    """Test getting initial supervoxel IDs"""
    mock_supervoxel_id = 99999

    def mock_get_segment_id(project_name, coordinate, initial, db_session):
        # Verify initial parameter is passed correctly
        assert initial is True
        return mock_supervoxel_id

    mocker.patch(
        "zetta_utils.task_management.segment.get_segment_id",
        side_effect=mock_get_segment_id,
    )

    coordinates = [[100.0, 200.0, 300.0]]
    result = {}

    for coordinate in coordinates:
        segment_id = get_segment_id(
            project_name=project_name,
            coordinate=coordinate,
            initial=True,
            db_session=db_session,
        )
        if segment_id != 0:
            result[segment_id] = coordinate

    assert len(result) == 1
    assert result[mock_supervoxel_id] == [100.0, 200.0, 300.0]


def test_get_segment_id_empty_coordinates(existing_project, db_session, project_name):
    """Test with empty coordinates list"""
    coordinates = []
    result = {}

    for coordinate in coordinates:
        segment_id = get_segment_id(
            project_name=project_name,
            coordinate=coordinate,
            initial=False,
            db_session=db_session,
        )
        if segment_id != 0:
            result[segment_id] = coordinate

    assert len(result) == 0
    assert not result


def test_get_segment_id_handles_duplicates(existing_project, db_session, project_name, mocker):
    """Test that duplicate segment IDs are handled (later coordinate overwrites earlier)"""
    # Mock: different coordinates map to same segment_id
    def mock_get_segment_id(project_name, coordinate, initial, db_session):
        # All coordinates return same segment_id
        return 12345

    mocker.patch(
        "zetta_utils.task_management.segment.get_segment_id",
        side_effect=mock_get_segment_id,
    )

    coordinates = [[100.0, 200.0, 300.0], [150.0, 250.0, 350.0], [200.0, 300.0, 400.0]]
    result = {}

    for coordinate in coordinates:
        segment_id = get_segment_id(
            project_name=project_name,
            coordinate=coordinate,
            initial=False,
            db_session=db_session,
        )
        if segment_id != 0:
            result[segment_id] = coordinate

    # Only one entry should exist (last coordinate overwrites)
    assert len(result) == 1
    assert result[12345] == [200.0, 300.0, 400.0]
