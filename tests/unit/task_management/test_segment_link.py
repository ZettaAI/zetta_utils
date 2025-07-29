"""Tests for segment_link module"""

# pylint: disable=unused-argument,redefined-outer-name

import contextlib
import json
import urllib.parse
from datetime import datetime, timezone

import pytest

from zetta_utils.task_management.db.models import EndpointModel, SegmentModel
from zetta_utils.task_management.project import create_project
from zetta_utils.task_management.segment_link import (
    get_segment_link,
    get_segment_ng_state,
)


@pytest.fixture
def existing_project_with_extra_layers(clean_db, db_session, project_name):
    """Create a project with extra layers"""
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Update the project to have extra_layers as a list (bypassing type check)
    # pylint: disable=import-outside-toplevel
    from zetta_utils.task_management.db.models import ProjectModel

    project = db_session.query(ProjectModel).filter_by(project_name=project_name).first()
    project.extra_layers = [
        {
            "type": "image",
            "source": "precomputed://gs://test-bucket/image",
            "name": "EM",
        }
    ]
    db_session.commit()

    yield project_name


@pytest.fixture
def existing_segment_with_root(clean_db, db_session, project_name):
    """Create a segment with root location"""
    # First create the project
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        root_x=150.0,
        root_y=250.0,
        root_z=350.0,
        current_segment_id=67890,
        task_ids=[],
        status="WIP",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()
    return segment


@pytest.fixture
def existing_endpoints(existing_segment_with_root, db_session, project_name):
    """Create various endpoints for a segment"""
    endpoints = [
        EndpointModel(
            project_name=project_name,
            seed_id=12345,
            endpoint_id=1,
            x=110.0,
            y=210.0,
            z=310.0,
            status="CERTAIN",
            user="user1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        EndpointModel(
            project_name=project_name,
            seed_id=12345,
            endpoint_id=2,
            x=120.0,
            y=220.0,
            z=320.0,
            status="UNCERTAIN",
            user="user1",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        EndpointModel(
            project_name=project_name,
            seed_id=12345,
            endpoint_id=3,
            x=130.0,
            y=230.0,
            z=330.0,
            status="BREADCRUMB",
            user="user2",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        # Add one more certain endpoint
        EndpointModel(
            project_name=project_name,
            seed_id=12345,
            endpoint_id=4,
            x=140.0,
            y=240.0,
            z=340.0,
            status="CERTAIN",
            user="user2",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    ]

    for endpoint in endpoints:
        db_session.add(endpoint)
    db_session.commit()

    return endpoints


def test_get_segment_ng_state_basic(existing_segment_with_root, db_session, project_name):
    """Test generating neuroglancer state for segment without endpoints"""
    ng_state = get_segment_ng_state(
        project_name=project_name,
        seed_id=12345,
        include_certain_ends=False,
        include_uncertain_ends=False,
        include_breadcrumbs=False,
        db_session=db_session,
    )

    # Check basic structure
    assert "dimensions" in ng_state
    assert "position" in ng_state
    assert ng_state["position"] == [100.0, 200.0, 300.0]  # seed position
    assert "layers" in ng_state

    # Check segmentation layer
    seg_layer = next(l for l in ng_state["layers"] if l["name"] == "Segmentation")
    assert seg_layer["type"] == "segmentation"
    assert seg_layer["segments"] == ["67890"]  # current_segment_id

    # Check seed location layer
    seed_layer = next(l for l in ng_state["layers"] if l["name"] == "Seed Location")
    assert seed_layer["type"] == "annotation"
    assert seed_layer["annotationColor"] == "#ff00ff"  # Purple
    assert len(seed_layer["annotations"]) == 1
    assert seed_layer["annotations"][0]["point"] == [100.0, 200.0, 300.0]

    # Check root location layer
    root_layer = next(l for l in ng_state["layers"] if l["name"] == "Root Location")
    assert root_layer["type"] == "annotation"
    assert root_layer["annotationColor"] == "#00ff00"  # Green
    assert len(root_layer["annotations"]) == 1
    assert root_layer["annotations"][0]["point"] == [150.0, 250.0, 350.0]

    # Should not have endpoint layers
    layer_names = [l["name"] for l in ng_state["layers"]]
    assert "Certain Ends" not in layer_names
    assert "Uncertain Ends" not in layer_names
    assert "Breadcrumbs" not in layer_names


def test_get_segment_ng_state_with_endpoints(existing_endpoints, db_session, project_name):
    """Test generating neuroglancer state with endpoints included"""
    ng_state = get_segment_ng_state(
        project_name=project_name,
        seed_id=12345,
        include_certain_ends=True,
        include_uncertain_ends=True,
        include_breadcrumbs=True,
        db_session=db_session,
    )

    # Check endpoint layers
    certain_layer = next(l for l in ng_state["layers"] if l["name"] == "Certain Ends")
    assert certain_layer["annotationColor"] == "#ffff00"  # Yellow
    assert len(certain_layer["annotations"]) == 2  # Two certain endpoints

    uncertain_layer = next(l for l in ng_state["layers"] if l["name"] == "Uncertain Ends")
    assert uncertain_layer["annotationColor"] == "#ff0000"  # Red
    assert len(uncertain_layer["annotations"]) == 1  # One uncertain endpoint

    breadcrumb_layer = next(l for l in ng_state["layers"] if l["name"] == "Breadcrumbs")
    assert breadcrumb_layer["annotationColor"] == "#0400ff"  # Blue
    assert len(breadcrumb_layer["annotations"]) == 1  # One breadcrumb


def test_get_segment_ng_state_no_root(clean_db, db_session, project_name):
    """Test generating state for segment without root location"""
    # Create project
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Create segment without root
    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="WIP",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    ng_state = get_segment_ng_state(
        project_name=project_name,
        seed_id=12345,
        include_certain_ends=False,
        include_uncertain_ends=False,
        include_breadcrumbs=False,
        db_session=db_session,
    )

    # Root location layer should exist but be empty
    root_layer = next(l for l in ng_state["layers"] if l["name"] == "Root Location")
    assert root_layer["annotations"] == []


def test_get_segment_ng_state_no_current_segment(clean_db, db_session, project_name):
    """Test generating state for segment without current_segment_id"""
    # Create project
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Create segment without current_segment_id
    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=None,
        task_ids=[],
        status="WIP",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    ng_state = get_segment_ng_state(
        project_name=project_name,
        seed_id=12345,
        db_session=db_session,
    )

    # Segmentation layer should have empty segments list
    seg_layer = next(l for l in ng_state["layers"] if l["name"] == "Segmentation")
    assert seg_layer["segments"] == []


def test_get_segment_ng_state_with_extra_layers(
    existing_project_with_extra_layers, db_session, project_name
):
    """Test that extra layers from project are included"""
    # Create segment
    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="WIP",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    ng_state = get_segment_ng_state(
        project_name=project_name,
        seed_id=12345,
        db_session=db_session,
    )

    # First layer should be the extra layer
    assert ng_state["layers"][0]["type"] == "image"
    assert ng_state["layers"][0]["name"] == "EM"
    assert ng_state["layers"][0]["source"] == "precomputed://gs://test-bucket/image"


def test_get_segment_ng_state_segment_not_found(clean_db, db_session, project_name):
    """Test error when segment doesn't exist"""
    # Create project
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    with pytest.raises(ValueError, match="Segment 99999 not found"):
        get_segment_ng_state(
            project_name=project_name,
            seed_id=99999,
            db_session=db_session,
        )


def test_get_segment_ng_state_project_not_found(clean_db, db_session):
    """Test error when project doesn't exist"""
    # Create a segment without a project (shouldn't happen in practice)
    segment = SegmentModel(
        project_name="nonexistent",
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        task_ids=[],
        status="WIP",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    with pytest.raises(ValueError, match="Project nonexistent not found"):
        get_segment_ng_state(
            project_name="nonexistent",
            seed_id=12345,
            db_session=db_session,
        )


def test_get_segment_link(existing_segment_with_root, db_session, project_name):
    """Test generating spelunker link"""
    link = get_segment_link(
        project_name=project_name,
        seed_id=12345,
        include_certain_ends=False,
        include_uncertain_ends=False,
        include_breadcrumbs=False,
        db_session=db_session,
    )

    # Check it's a spelunker URL
    assert link.startswith("https://spelunker.cave-explorer.org/#!")

    # Extract and decode the state
    encoded_state = link.split("#!")[1]
    decoded_state = urllib.parse.unquote(encoded_state)
    ng_state = json.loads(decoded_state)

    # Verify it's a valid state
    assert "dimensions" in ng_state
    assert "layers" in ng_state
    assert ng_state["position"] == [100.0, 200.0, 300.0]


def test_get_segment_link_with_endpoints(existing_endpoints, db_session, project_name):
    """Test generating spelunker link with endpoints"""
    link = get_segment_link(
        project_name=project_name,
        seed_id=12345,
        include_certain_ends=True,
        include_uncertain_ends=True,
        include_breadcrumbs=True,
        db_session=db_session,
    )

    # Extract and decode the state
    encoded_state = link.split("#!")[1]
    decoded_state = urllib.parse.unquote(encoded_state)
    ng_state = json.loads(decoded_state)

    # Verify endpoint layers are included
    layer_names = [l["name"] for l in ng_state["layers"]]
    assert "Certain Ends" in layer_names
    assert "Uncertain Ends" in layer_names
    assert "Breadcrumbs" in layer_names


def test_get_segment_ng_state_no_session(
    existing_segment_with_root, db_session, project_name, mocker
):
    """Test getting segment ng state without providing a session"""

    # Mock get_session_context to return our test session
    def mock_get_session_context():
        @contextlib.contextmanager
        def context():
            try:
                yield db_session
            finally:
                pass

        return context()

    mocker.patch(
        "zetta_utils.task_management.segment_link.get_session_context",
        side_effect=mock_get_session_context,
    )

    # Call without db_session parameter to trigger the else branch
    state = get_segment_ng_state(project_name, 12345)

    assert "layers" in state
    assert "position" in state
    assert state["position"] == [100.0, 200.0, 300.0]


def test_get_segment_ng_state_with_segment_type_layers(clean_db, db_session, project_name, mocker):
    """Test generating neuroglancer state with segment type layers included"""
    # Create project
    create_project(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
        db_session=db_session,
    )

    # Create segment with expected_segment_type
    segment = SegmentModel(
        project_name=project_name,
        seed_id=12345,
        seed_x=100.0,
        seed_y=200.0,
        seed_z=300.0,
        current_segment_id=67890,
        expected_segment_type="neuron",  # This will trigger the segment type layers
        task_ids=[],
        status="WIP",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()

    # Mock get_segment_type_layers to return sample layers
    mock_type_layers = [
        {
            "type": "annotation",
            "name": "Neuron Reference",
            "annotationColor": "#00ff00",
            "annotations": [],
        },
        {
            "type": "segmentation",
            "name": "Neuron Sample",
            "source": "precomputed://example",
            "segments": ["123", "456"],
        },
    ]

    mock_get_segment_type_layers = mocker.patch(
        "zetta_utils.task_management.segment_link.get_segment_type_layers",
        return_value=mock_type_layers,
    )

    # Call with include_segment_type_layers=True (default)
    ng_state = get_segment_ng_state(
        project_name=project_name,
        seed_id=12345,
        include_segment_type_layers=True,
        db_session=db_session,
    )

    # Check that segment type layers were added
    layer_names = [l["name"] for l in ng_state["layers"]]
    assert "Neuron Reference" in layer_names
    assert "Neuron Sample" in layer_names

    # Verify the get_segment_type_layers was called with correct parameters
    mock_get_segment_type_layers.assert_called_once_with(
        project_name=project_name,
        segment_type_name="neuron",
        include_names=False,
        db_session=db_session,
    )
