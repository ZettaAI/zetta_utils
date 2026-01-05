"""Tests for segment module"""

# pylint: disable=unused-argument,redefined-outer-name

import os
import random
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from zetta_utils.task_management.db.models import ProjectModel, SegmentModel
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.segment import (
    convert_to_sv_resolution,
    create_segment_from_coordinate,
    get_segment_id,
    get_skeleton_length_mm,
    update_segment_statistics,
)


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


@pytest.fixture
def existing_segment(existing_project, db_session, project_name):
    """Create a segment in the database"""
    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()
    return segment


def test_get_segment_id_current(existing_project, db_session, project_name, mocker):
    """Test getting current agglomerated segment ID"""
    coordinate = [100.0, 200.0, 300.0]

    # Mock only the external CloudVolume layer
    mock_layer = mocker.Mock()
    # Set up the layer to return segment ID when sliced
    mock_layer.__getitem__ = mocker.Mock(return_value=np.array([[[67890]]], dtype=np.uint64))

    mocker.patch("zetta_utils.task_management.segment.build_cv_layer", return_value=mock_layer)

    segment_id = get_segment_id(project_name, coordinate, initial=False, db_session=db_session)

    assert segment_id == 67890


def test_get_segment_id_initial(existing_project, db_session, project_name, mocker):
    """Test getting initial supervoxel ID"""
    coordinate = [100.0, 200.0, 300.0]

    # Mock only the external CloudVolume layer
    mock_layer = mocker.Mock()
    mock_layer.__getitem__ = mocker.Mock(return_value=np.array([[[12345]]], dtype=np.uint64))

    mocker.patch("zetta_utils.task_management.segment.build_cv_layer", return_value=mock_layer)

    segment_id = get_segment_id(project_name, coordinate, initial=True, db_session=db_session)

    assert segment_id == 12345


def test_get_segment_id_project_not_found(db_session):
    """Test getting segment ID when project doesn't exist"""
    with pytest.raises(ValueError, match="Project 'nonexistent' not found"):
        get_segment_id("nonexistent", [100.0, 200.0, 300.0], db_session=db_session)


def test_convert_to_sv_resolution():
    """Test coordinate conversion to supervoxel resolution"""
    coordinate = [100.0, 200.0, 10.0]
    from_resolution = [16, 16, 80]
    sv_resolution = [8, 8, 40]

    result = convert_to_sv_resolution(coordinate, from_resolution, sv_resolution)

    assert result == [200.0, 400.0, 20.0]  # Each coordinate is scaled by 2x


def test_get_skeleton_length_mm_success(existing_project, db_session, project_name, mocker):
    """Test successfully getting skeleton length"""
    segment_id = 67890

    # Set up the auth token
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock the external dependencies
    mock_skeleton = mocker.Mock()
    mock_skeleton.path_length.return_value = 1500000  # 1.5 mm (1,500,000 nm)

    mocker.patch(
        "zetta_utils.task_management.segment.pcg_skel.pcg_skeleton", return_value=mock_skeleton
    )
    mocker.patch("zetta_utils.task_management.segment.CAVEclient")

    length_mm = get_skeleton_length_mm(project_name, segment_id, db_session=db_session)

    assert length_mm == 1.5


def test_get_skeleton_length_mm_no_datastack(clean_db, db_session, project_name):
    """Test getting skeleton length when project has no datastack"""
    # Create project without datastack_name
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    with pytest.raises(ValueError, match="does not have a datastack_name configured"):
        get_skeleton_length_mm(project_name, 67890, db_session=db_session)


def test_get_skeleton_length_mm_exception(existing_project, db_session, project_name, mocker):
    """Test handling exception when getting skeleton fails"""
    # Set up the auth token
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    mocker.patch(
        "zetta_utils.task_management.segment.CAVEclient",
        side_effect=Exception("Connection failed"),
    )

    length_mm = get_skeleton_length_mm(project_name, 67890, db_session=db_session)

    assert length_mm is None


def test_update_segment_statistics_success(
    existing_segment, existing_project, db_session, project_name, mocker
):
    """Test successfully updating segment statistics"""
    seed_id = 12345

    # Set up auth token
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock get_segment_id to return the current segment ID (simulating no change)
    mocker.patch("zetta_utils.task_management.segment.get_segment_id", return_value=67890)

    # Mock skeleton
    mock_skeleton = mocker.Mock()
    mock_skeleton.path_length.return_value = 2500000  # 2.5 mm (2,500,000 nm)
    mocker.patch(
        "zetta_utils.task_management.segment.pcg_skel.pcg_skeleton", return_value=mock_skeleton
    )

    # Mock synapse queries using live_query
    mock_client = mocker.Mock()
    mock_client.materialize.live_query.side_effect = [
        [],
        [],
    ]
    mocker.patch("zetta_utils.task_management.segment.CAVEclient", return_value=mock_client)

    result = update_segment_statistics(project_name, seed_id, db_session=db_session)

    assert result == {
        "skeleton_path_length_mm": 2.5,
        "pre_synapse_count": 0,
        "post_synapse_count": 0,
    }


def test_update_segment_statistics_no_current_segment_id(
    existing_project, db_session, project_name, mocker
):
    """Test updating statistics when segment has no current_segment_id"""
    # Mock get_segment_id to return None (no segment found at seed location)
    mocker.patch("zetta_utils.task_management.segment.get_segment_id", return_value=None)

    # Create a segment with no current_segment_id in the test database
    segment = SegmentModel(
        project_name=project_name,
        seed_id=99999,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=None,  # No current segment ID
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    # Mock environment variables to prevent CAVEclient from writing config files
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock CAVEclient to prevent config file creation (though it shouldn't be called)
    mocker.patch("zetta_utils.task_management.segment.CAVEclient")

    result = update_segment_statistics(project_name, 99999, db_session=db_session)

    assert result == {"error": "No segment found at seed location"}


def test_update_segment_statistics_skeleton_error(
    existing_segment, existing_project, db_session, project_name, mocker
):
    """Test handling skeleton computation error"""
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock get_segment_id to return the current segment ID
    mocker.patch("zetta_utils.task_management.segment.get_segment_id", return_value=67890)

    # Mock the database queries
    mock_segment = mocker.Mock()
    mock_segment.current_segment_id = 67890

    mock_project = mocker.Mock()
    mock_project.datastack_name = "test_datastack"
    mock_project.synapse_table = "test_synapse_table"

    mock_session = mocker.Mock()
    mock_session.query.return_value.filter_by.return_value.first.side_effect = [
        mock_segment,
        mock_project,
    ]
    mock_session.commit = mocker.Mock()

    mocker.patch(
        "zetta_utils.task_management.segment.get_session_context"
    ).__enter__.return_value = mock_session

    # Mock skeleton failure
    mocker.patch(
        "zetta_utils.task_management.segment.pcg_skel.pcg_skeleton",
        side_effect=Exception("Skeleton computation failed"),
    )

    # Mock synapse queries success
    mock_client = mocker.Mock()
    mock_client.materialize.live_query.side_effect = [
        [{"id": 1}],  # 1 pre-synapse
        [],  # 0 post-synapses
    ]
    mocker.patch("zetta_utils.task_management.segment.CAVEclient", return_value=mock_client)

    result = update_segment_statistics(project_name, 12345, db_session=db_session)

    assert "skeleton_error" in result
    assert result["pre_synapse_count"] == 1
    assert result["post_synapse_count"] == 0


def test_update_segment_statistics_synapse_error(
    existing_segment, existing_project, db_session, project_name, mocker
):
    """Test handling synapse query error"""
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock get_segment_id to return the current segment ID
    mocker.patch("zetta_utils.task_management.segment.get_segment_id", return_value=67890)

    # Mock the database queries
    mock_segment = mocker.Mock()
    mock_segment.current_segment_id = 67890

    mock_project = mocker.Mock()
    mock_project.datastack_name = "test_datastack"
    mock_project.synapse_table = "test_synapse_table"

    mock_session = mocker.Mock()
    mock_session.query.return_value.filter_by.return_value.first.side_effect = [
        mock_segment,
        mock_project,
    ]
    mock_session.commit = mocker.Mock()

    mocker.patch(
        "zetta_utils.task_management.segment.get_session_context"
    ).__enter__.return_value = mock_session

    # Mock skeleton success
    mock_skeleton = mocker.Mock()
    mock_skeleton.path_length.return_value = 1000000  # 1.0 mm (1,000,000 nm)
    mocker.patch(
        "zetta_utils.task_management.segment.pcg_skel.pcg_skeleton", return_value=mock_skeleton
    )

    # Mock synapse query failure
    mock_client = mocker.Mock()
    mock_client.materialize.live_query.side_effect = Exception("Query failed")
    mocker.patch("zetta_utils.task_management.segment.CAVEclient", return_value=mock_client)

    result = update_segment_statistics(project_name, 12345, db_session=db_session)

    assert result["skeleton_path_length_mm"] == 1.0
    assert "synapse_error" in result


def test_update_segment_statistics_segment_not_found(
    existing_project, db_session, project_name, mocker
):
    """Test updating statistics for non-existent segment"""
    # Mock environment variables to prevent CAVEclient from writing config files
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock CAVEclient to prevent config file creation (though it shouldn't be called)
    mocker.patch("zetta_utils.task_management.segment.CAVEclient")

    with pytest.raises(ValueError, match="Segment with seed_id 99999 not found"):
        update_segment_statistics(project_name, 99999, db_session=db_session)


def test_update_segment_statistics_no_synapse_table(clean_db, db_session, project_name, mocker):
    """Test updating statistics when project has no synapse_table"""
    # Create project without synapse_table
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        datastack_name="test_datastack",
        # No synapse_table parameter
        db_session=db_session,
    )

    # Create a segment
    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    # Mock environment variables to prevent CAVEclient from writing config files
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock CAVEclient to prevent config file creation (though it shouldn't be called)
    mocker.patch("zetta_utils.task_management.segment.CAVEclient")

    with pytest.raises(ValueError, match="does not have a synapse_table configured"):
        update_segment_statistics(project_name, 12345, db_session=db_session)


def test_create_segment_from_coordinate_success(
    existing_project, db_session, project_name, mocker
):
    """Test successfully creating a segment from coordinate"""
    coordinate = [100.0, 200.0, 300.0]

    # Use a unique seed ID that doesn't exist in the database
    new_seed_id = random.randint(1000000, 9999999)

    # No need to mock session anymore

    # Mock the volume layer to return segment IDs
    mock_layer1 = mocker.Mock()
    mock_layer1.__getitem__ = mocker.Mock(
        return_value=np.array([[[new_seed_id]]], dtype=np.uint64)
    )
    mock_layer2 = mocker.Mock()
    mock_layer2.__getitem__ = mocker.Mock(return_value=np.array([[[67890]]], dtype=np.uint64))

    mocker.patch(
        "zetta_utils.task_management.segment.build_cv_layer",
        side_effect=[mock_layer1, mock_layer2],
    )

    segment = create_segment_from_coordinate(
        project_name=project_name,
        coordinate=coordinate,
        batch="batch_1",
        segment_type="type_A",
        expected_segment_type="type_A",
        extra_data={"notes": "test segment"},
        db_session=db_session,
    )

    assert segment["seed_id"] == new_seed_id
    assert segment["seed_x"] == 100.0
    assert segment["seed_y"] == 200.0
    assert segment["seed_z"] == 300.0
    assert segment["current_segment_id"] == 67890
    assert segment["batch"] == "batch_1"
    assert segment["segment_type"] == "type_A"
    assert segment["expected_segment_type"] == "type_A"
    assert segment["extra_data"] == {"notes": "test segment"}
    assert segment["status"] == "Raw"
    assert segment["is_exported"] is False
    assert not segment["task_ids"]

    # Verify segment was saved to database
    saved_segment = (
        db_session.query(SegmentModel)
        .filter_by(project_name=project_name, seed_id=new_seed_id)
        .first()
    )
    assert saved_segment is not None
    assert saved_segment.current_segment_id == 67890


def test_create_segment_from_coordinate_no_supervoxel(
    existing_project, db_session, project_name, mocker
):
    """Test creating segment when no supervoxel found at coordinate"""
    coordinate = [100.0, 200.0, 300.0]

    # Mock layer to return 0 (no supervoxel)
    mock_layer = mocker.Mock()
    mock_layer.__getitem__ = mocker.Mock(return_value=np.array([[[0]]], dtype=np.uint64))
    mocker.patch("zetta_utils.task_management.segment.build_cv_layer", return_value=mock_layer)

    with pytest.raises(ValueError, match="No supervoxel found at coordinate"):
        create_segment_from_coordinate(project_name, coordinate, db_session=db_session)


def test_create_segment_from_coordinate_already_exists(
    existing_segment, existing_project, db_session, project_name, mocker
):
    """Test creating segment when it already exists"""
    coordinate = [100.0, 200.0, 300.0]

    # Mock layer to return the same seed_id as existing segment
    mock_layer1 = mocker.Mock()
    mock_layer1.__getitem__ = mocker.Mock(return_value=np.array([[[12345]]], dtype=np.uint64))
    mock_layer2 = mocker.Mock()
    mock_layer2.__getitem__ = mocker.Mock(return_value=np.array([[[67890]]], dtype=np.uint64))

    mocker.patch(
        "zetta_utils.task_management.segment.build_cv_layer",
        side_effect=[mock_layer1, mock_layer2],
    )

    with pytest.raises(ValueError, match="Segment with seed_id 12345 already exists"):
        create_segment_from_coordinate(project_name, coordinate, db_session=db_session)


def test_create_segment_from_coordinate_no_current_segment(
    existing_project, db_session, project_name, mocker
):
    """Test creating segment when current_segment_id is 0"""
    coordinate = [100.0, 200.0, 300.0]

    # Use unique seed ID
    new_seed_id = random.randint(10000000, 99999999)

    # No need to mock session anymore

    # Mock layer to return 0 for current_segment_id
    mock_layer1 = mocker.Mock()
    mock_layer1.__getitem__ = mocker.Mock(
        return_value=np.array([[[new_seed_id]]], dtype=np.uint64)
    )
    mock_layer2 = mocker.Mock()
    mock_layer2.__getitem__ = mocker.Mock(return_value=np.array([[[0]]], dtype=np.uint64))

    mocker.patch(
        "zetta_utils.task_management.segment.build_cv_layer",
        side_effect=[mock_layer1, mock_layer2],
    )

    segment = create_segment_from_coordinate(project_name, coordinate, db_session=db_session)

    assert segment["seed_id"] == new_seed_id
    assert segment["current_segment_id"] is None  # Should be None when 0

    # Verify in database
    saved_segment = (
        db_session.query(SegmentModel)
        .filter_by(project_name=project_name, seed_id=new_seed_id)
        .first()
    )
    assert saved_segment.current_segment_id is None


def test_create_segment_from_coordinate_minimal(clean_db, db_session, project_name, mocker):
    """Test creating segment with minimal parameters"""
    # First create the project
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    coordinate = [100.0, 200.0, 300.0]

    # Use a very large unique seed ID based on test run
    new_seed_id = int(str(uuid.uuid4().int)[:9])

    # Mock layer
    mock_layer1 = mocker.Mock()
    mock_layer1.__getitem__ = mocker.Mock(
        return_value=np.array([[[new_seed_id]]], dtype=np.uint64)
    )
    mock_layer2 = mocker.Mock()
    mock_layer2.__getitem__ = mocker.Mock(return_value=np.array([[[67890]]], dtype=np.uint64))

    mocker.patch(
        "zetta_utils.task_management.segment.build_cv_layer",
        side_effect=[mock_layer1, mock_layer2],
    )

    segment = create_segment_from_coordinate(project_name, coordinate, db_session=db_session)

    # Verify required fields are present
    assert segment["seed_id"] == new_seed_id
    assert segment["project_name"] == project_name
    assert segment["status"] == "Raw"
    assert segment["is_exported"] is False

    # Verify optional fields are not present
    assert "batch" not in segment
    assert "segment_type" not in segment
    assert "expected_segment_type" not in segment
    assert "extra_data" not in segment


def test_get_skeleton_length_mm_project_not_found(db_session, mocker):
    """Test getting skeleton length when project doesn't exist"""
    # The project won't exist in our test database
    with pytest.raises(ValueError, match="Project 'nonexistent' not found!"):
        get_skeleton_length_mm("nonexistent", 12345, db_session=db_session)


def test_update_segment_statistics_project_not_found(db_session, project_name, mocker):
    """Test updating statistics when project doesn't exist"""
    # Mock environment variables
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Create a segment but no project (simulating a bad state)
    segment = SegmentModel(
        project_name="nonexistent_project",
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    with pytest.raises(ValueError, match="Project 'nonexistent_project' not found!"):
        update_segment_statistics("nonexistent_project", 12345, db_session=db_session)


def test_update_segment_statistics_no_datastack_name(
    existing_project, db_session, project_name, mocker
):
    """Test updating statistics when project has no datastack_name"""
    # Mock get_segment_id (though it won't be called due to early exit)
    mocker.patch("zetta_utils.task_management.segment.get_segment_id", return_value=67890)

    # Create a segment
    segment = SegmentModel(
        project_name=project_name,
        seed_id=88888,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="Raw",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)

    # Remove datastack_name from project
    project = db_session.query(ProjectModel).filter_by(project_name=project_name).first()
    project.datastack_name = None
    db_session.commit()

    # Mock environment variables
    mocker.patch.dict(os.environ, {"CAVE_AUTH_TOKEN": "test_token"})

    # Mock CAVEclient to prevent config file creation
    mocker.patch("zetta_utils.task_management.segment.CAVEclient")

    with pytest.raises(ValueError, match="does not have a datastack_name configured!"):
        update_segment_statistics(project_name, 88888, db_session=db_session)
