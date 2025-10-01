"""Tests for ingest segments API endpoint"""

# pylint: disable=unused-argument,redefined-outer-name

import pytest

from zetta_utils.task_management.db.models import SegmentTypeModel
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.seg_trace_utils.ingest_segment_coordinates import (
    ingest_validated_coordinates,
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
def existing_segment_type(existing_project, db_session, project_name):
    """Create a segment type in the database"""
    segment_type = SegmentTypeModel(
        project_name=project_name,
        type_name="axon",
        reference_segment=12345,
    )
    db_session.add(segment_type)
    db_session.commit()
    return segment_type


def test_ingest_segments_direct_function_success(
    existing_project, existing_segment_type, db_session, project_name, mocker
):
    """Test the ingest_validated_coordinates function directly"""
    # Mock the segment creation
    mock_segment = {
        "seed_id": 12345,
        "seed_x": 100.0,
        "seed_y": 200.0,
        "seed_z": 300.0,
        "current_segment_id": 67890,
        "batch": "test_batch",
        "expected_segment_type": "axon",
        "status": "WIP",
        "is_exported": False,
        "task_ids": [],
    }

    mocker.patch(
        "zetta_utils.task_management.seg_trace_utils.ingest_segment_coordinates.create_segment_from_coordinate",
        return_value=mock_segment,
    )

    valid_coordinates = [
        {"coordinate": [100.0, 200.0, 300.0], "segment_id": 67890},
        {"coordinate": [150.0, 250.0, 350.0], "segment_id": 67891},
    ]

    results = ingest_validated_coordinates(
        project_name=project_name,
        valid_coordinates=valid_coordinates,
        expected_neuron_type="axon",
        batch_name="test_batch",
        db_session=db_session,
    )

    assert results["created_segments"] == 2
    assert results["creation_errors"] == 0
    assert len(results["created_seed_ids"]) == 2
    assert results["created_seed_ids"] == [12345, 12345]


def test_ingest_segments_empty_coordinates(
    existing_project, existing_segment_type, db_session, project_name
):
    """Test ingesting with empty coordinates list"""
    results = ingest_validated_coordinates(
        project_name=project_name,
        valid_coordinates=[],
        expected_neuron_type="axon",
        batch_name="test_batch",
        db_session=db_session,
    )

    assert results["created_segments"] == 0
    assert results["creation_errors"] == 0
    assert len(results["created_seed_ids"]) == 0


def test_ingest_segments_with_different_neuron_types(
    existing_project, db_session, project_name, mocker
):
    """Test ingesting segments with multiple neuron types"""
    # Create multiple segment types
    segment_type1 = SegmentTypeModel(
        project_name=project_name,
        type_name="axon",
        reference_segment=12345,
    )
    segment_type2 = SegmentTypeModel(
        project_name=project_name,
        type_name="dendrite",
        reference_segment=54321,
    )
    db_session.add(segment_type1)
    db_session.add(segment_type2)
    db_session.commit()

    # Mock the segment creation
    mock_segment1 = {
        "seed_id": 12345,
        "seed_x": 100.0,
        "seed_y": 200.0,
        "seed_z": 300.0,
        "current_segment_id": 67890,
        "batch": "test_batch_axon",
        "expected_segment_type": "axon",
        "status": "WIP",
        "is_exported": False,
        "task_ids": [],
    }
    mock_segment2 = {
        "seed_id": 54321,
        "seed_x": 150.0,
        "seed_y": 250.0,
        "seed_z": 350.0,
        "current_segment_id": 67891,
        "batch": "test_batch_dendrite",
        "expected_segment_type": "dendrite",
        "status": "WIP",
        "is_exported": False,
        "task_ids": [],
    }

    mocker.patch(
        "zetta_utils.task_management.seg_trace_utils.ingest_segment_coordinates.create_segment_from_coordinate",
        side_effect=[mock_segment1, mock_segment2],
    )

    # Test axon ingestion
    valid_coordinates_axon = [{"coordinate": [100.0, 200.0, 300.0], "segment_id": 67890}]

    results_axon = ingest_validated_coordinates(
        project_name=project_name,
        valid_coordinates=valid_coordinates_axon,
        expected_neuron_type="axon",
        batch_name="test_batch_axon",
        db_session=db_session,
    )

    assert results_axon["created_segments"] == 1
    assert results_axon["created_seed_ids"] == [12345]

    # Test dendrite ingestion
    valid_coordinates_dendrite = [{"coordinate": [150.0, 250.0, 350.0], "segment_id": 67891}]

    results_dendrite = ingest_validated_coordinates(
        project_name=project_name,
        valid_coordinates=valid_coordinates_dendrite,
        expected_neuron_type="dendrite",
        batch_name="test_batch_dendrite",
        db_session=db_session,
    )

    assert results_dendrite["created_segments"] == 1
    assert results_dendrite["created_seed_ids"] == [54321]


def test_ingest_segments_progress_logging(
    existing_project, existing_segment_type, db_session, project_name, mocker
):
    """Test that progress logging works for large batches"""
    # Mock the segment creation
    mock_segment = {
        "seed_id": 12345,
        "seed_x": 100.0,
        "seed_y": 200.0,
        "seed_z": 300.0,
        "current_segment_id": 67890,
        "batch": "test_batch",
        "expected_segment_type": "axon",
        "status": "WIP",
        "is_exported": False,
        "task_ids": [],
    }

    mocker.patch(
        "zetta_utils.task_management.seg_trace_utils.ingest_segment_coordinates.create_segment_from_coordinate",
        return_value=mock_segment,
    )

    # Create 150 coordinates to trigger progress logging (every 100)
    valid_coordinates = [
        {"coordinate": [i * 10.0, i * 20.0, i * 30.0], "segment_id": 67890 + i}
        for i in range(150)
    ]

    results = ingest_validated_coordinates(
        project_name=project_name,
        valid_coordinates=valid_coordinates,
        expected_neuron_type="axon",
        batch_name="test_batch",
        db_session=db_session,
    )

    assert results["created_segments"] == 150
    assert results["creation_errors"] == 0
    assert len(results["created_seed_ids"]) == 150