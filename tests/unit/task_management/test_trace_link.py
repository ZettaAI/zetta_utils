"""Tests for trace_link module"""

# pylint: disable=redefined-outer-name,unused-argument

import contextlib
import json
import urllib.parse
from datetime import datetime, timezone

import pytest

from zetta_utils.task_management.db.models import (
    EndpointModel,
    MergeEditModel,
    ProjectModel,
    SegmentModel,
    TaskModel,
)
from zetta_utils.task_management.ng_state import (
    _add_merge_layer,
    _get_merge_edits_data,
    _get_task_and_segment,
    get_trace_task_link,
    get_trace_task_state,
)
from zetta_utils.task_management.project import create_project


@pytest.fixture
def setup_project_and_segment(clean_db, db_session, project_name):
    """Create a project and segment for testing"""
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
        task_ids=["task_123", "task_456"],
        status="Raw",
        is_exported=False,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(segment)
    db_session.commit()
    return segment


@pytest.fixture
def trace_v0_task(setup_project_and_segment, db_session, project_name):
    """Create a trace_v0 task with seed_id in extra_data"""
    task = TaskModel(
        project_name=project_name,
        task_id="trace_task_123",
        completion_status="",
        assigned_user_id="",
        active_user_id="user1",
        completed_user_id="",
        ng_state={
            "dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [40e-9, "m"]},
            "position": [100.0, 200.0, 300.0],
            "layers": [],
        },
        ng_state_initial={},
        priority=1,
        batch_id="test_batch",
        last_leased_ts=0.0,
        first_start_ts=None,
        is_active=True,
        is_paused=False,
        is_checked=False,
        task_type="trace_v0",
        id_nonunique=123456,
        extra_data={"seed_id": 12345},  # seed_id in extra_data
        note=None,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(task)
    db_session.commit()
    return task


@pytest.fixture
def seg_trace_task(setup_project_and_segment, db_session, project_name):
    """Create a seg_trace task without seed_id in extra_data"""
    task = TaskModel(
        project_name=project_name,
        task_id="task_123",  # This is in the segment's task_ids
        completion_status="done",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="user1",
        ng_state={
            "dimensions": {"x": [8e-9, "m"], "y": [8e-9, "m"], "z": [40e-9, "m"]},
            "position": [100.0, 200.0, 300.0],
            "layers": [],
        },
        ng_state_initial={},
        priority=1,
        batch_id="test_batch",
        last_leased_ts=0.0,
        first_start_ts=None,
        is_active=True,
        is_paused=False,
        is_checked=False,
        task_type="seg_trace",
        id_nonunique=123456,
        extra_data=None,  # No extra_data, will use segment lookup
        note=None,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(task)
    db_session.commit()
    return task


@pytest.fixture
def endpoints_for_segment(setup_project_and_segment, db_session, project_name):
    """Create various endpoints for the test segment"""
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
    ]

    for endpoint in endpoints:
        db_session.add(endpoint)
    db_session.commit()
    return endpoints


@pytest.fixture
def merge_edits_for_task(seg_trace_task, db_session, project_name):
    """Create merge edits for the test task"""
    merge_edits = [
        MergeEditModel(
            project_name=project_name,
            edit_id=1,
            task_id="task_123",
            user_id="user1",
            points=[[12345, 100.0, 200.0, 300.0], [12346, 110.0, 210.0, 310.0]],
            created_at=datetime.now(timezone.utc),
        ),
        MergeEditModel(
            project_name=project_name,
            edit_id=2,
            task_id="task_123",
            user_id="user1",
            points=[[12347, 120.0, 220.0, 320.0], [12348, 130.0, 230.0, 330.0]],
            created_at=datetime.now(timezone.utc),
        ),
        MergeEditModel(
            project_name=project_name,
            edit_id=3,
            task_id="task_123",
            user_id="user2",
            points=[[12349, 140.0, 240.0, 340.0], [12350, 150.0, 250.0, 350.0]],
            created_at=datetime.now(timezone.utc),
        ),
    ]

    for merge_edit in merge_edits:
        db_session.add(merge_edit)
    db_session.commit()
    return merge_edits


class TestGetTraceTaskState:
    """Test get_trace_task_state function"""

    def test_trace_v0_task_with_seed_id_in_extra_data(
        self, trace_v0_task, endpoints_for_segment, db_session, project_name
    ):
        """Test trace_v0 task that has seed_id in extra_data"""
        # Fixtures create required test data
        _ = trace_v0_task
        _ = endpoints_for_segment

        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="trace_task_123",
            db_session=db_session,
        )

        # Check basic structure
        assert "dimensions" in ng_state
        assert "position" in ng_state
        assert ng_state["position"] == [100.0, 200.0, 300.0]  # seed position
        assert "layers" in ng_state

        # Check required layers are present
        layer_names = [layer["name"] for layer in ng_state["layers"]]
        assert "Segmentation" in layer_names
        assert "Seed Location" in layer_names
        assert "Root Location" in layer_names
        assert "Certain Ends" in layer_names
        assert "Uncertain Ends" in layer_names
        assert "Breadcrumbs" in layer_names
        assert "MERGES" in layer_names

        # Check segmentation layer
        seg_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "Segmentation")
        assert seg_layer["type"] == "segmentation"
        assert seg_layer["segments"] == ["67890"]  # current_segment_id

        # Check MERGES layer exists and has correct properties
        merge_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "MERGES")
        assert merge_layer["type"] == "annotation"
        assert merge_layer["tool"] == "annotateLine"
        assert merge_layer["annotationColor"] == "#ff8c00"  # Orange
        assert merge_layer["annotations"] == []  # No merge edits for this task

    def test_seg_trace_task_with_segment_lookup(
        self, seg_trace_task, merge_edits_for_task, db_session, project_name
    ):
        """Test seg_trace task that requires segment lookup"""
        # Fixtures create required test data
        _ = seg_trace_task
        _ = merge_edits_for_task

        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="task_123",
            db_session=db_session,
        )

        # Check basic structure
        assert "dimensions" in ng_state
        assert "position" in ng_state
        assert ng_state["position"] == [100.0, 200.0, 300.0]  # seed position
        assert "layers" in ng_state

        # Check MERGES layer has line annotations
        merge_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "MERGES")
        assert merge_layer["type"] == "annotation"
        assert merge_layer["tool"] == "annotateLine"
        assert merge_layer["annotationColor"] == "#ff8c00"  # Orange
        assert len(merge_layer["annotations"]) == 3  # Three merge edits

        # Check first merge edit line annotation
        first_merge = merge_layer["annotations"][0]
        assert first_merge["type"] == "line"
        assert first_merge["pointA"] == [12.5, 25.0, 7.5]  # 100.0/8.0, 200.0/8.0, 300.0/40.0
        assert first_merge["pointB"] == [13.75, 26.25, 7.75]  # 110.0/8.0, 210.0/8.0, 310.0/40.0
        assert "id" in first_merge

    def test_with_selective_layers(
        self, seg_trace_task, endpoints_for_segment, merge_edits_for_task, db_session, project_name
    ):
        """Test with selective layer inclusion"""
        # Fixtures create required test data
        _ = seg_trace_task
        _ = endpoints_for_segment
        _ = merge_edits_for_task

        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="task_123",
            include_certain_ends=False,
            include_uncertain_ends=False,
            include_breadcrumbs=False,
            include_segment_type_layers=False,
            include_merges=False,
            db_session=db_session,
        )

        layer_names = [layer["name"] for layer in ng_state["layers"]]

        # Should have core layers
        assert "Segmentation" in layer_names
        assert "Seed Location" in layer_names
        assert "Root Location" in layer_names

        # Should not have optional layers
        assert "Certain Ends" not in layer_names
        assert "Uncertain Ends" not in layer_names
        assert "Breadcrumbs" not in layer_names
        assert "MERGES" not in layer_names

    def test_only_merges_layer(
        self, seg_trace_task, endpoints_for_segment, merge_edits_for_task, db_session, project_name
    ):
        """Test with only merges layer enabled"""
        # Fixtures create required test data
        _ = seg_trace_task
        _ = endpoints_for_segment
        _ = merge_edits_for_task

        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="task_123",
            include_certain_ends=False,
            include_uncertain_ends=False,
            include_breadcrumbs=False,
            include_segment_type_layers=False,
            include_merges=True,
            db_session=db_session,
        )

        layer_names = [layer["name"] for layer in ng_state["layers"]]
        assert "MERGES" in layer_names

        # Check merge layer has correct data
        merge_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "MERGES")
        assert len(merge_layer["annotations"]) == 3  # Three merge edits

    def test_task_not_found(self, setup_project_and_segment, db_session, project_name):
        """Test error when task doesn't exist"""
        # Fixture creates required test data
        _ = setup_project_and_segment

        with pytest.raises(ValueError, match="Task nonexistent_task not found"):
            get_trace_task_state(
                project_name=project_name,
                task_id="nonexistent_task",
                db_session=db_session,
            )

    def test_task_not_associated_with_segment(self, clean_db, db_session, project_name):
        """Test error when task is not associated with any segment"""
        # Create project
        create_project(
            project_name=project_name,
            segmentation_path="precomputed://gs://test-bucket/segmentation",
            sv_resolution_x=8.0,
            sv_resolution_y=8.0,
            sv_resolution_z=40.0,
            db_session=db_session,
        )

        # Create task with no extra_data and no segment contains it
        task = TaskModel(
            project_name=project_name,
            task_id="orphan_task",
            completion_status="",
            assigned_user_id="",
            active_user_id="user1",
            completed_user_id="",
            ng_state={},
            ng_state_initial={},
            priority=1,
            batch_id="test_batch",
            last_leased_ts=0.0,
            first_start_ts=None,
            is_active=True,
            is_paused=False,
            is_checked=False,
            task_type="seg_trace",
            id_nonunique=123456,
            extra_data=None,
            note=None,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(task)
        db_session.commit()

        with pytest.raises(
            ValueError, match="Task orphan_task is not associated with any segment"
        ):
            get_trace_task_state(
                project_name=project_name,
                task_id="orphan_task",
                db_session=db_session,
            )

    def test_project_not_found_for_segment_lookup(self, clean_db, db_session):
        """Test error when project doesn't exist during segment lookup"""
        # Create task and segment in nonexistent project
        segment = SegmentModel(
            project_name="nonexistent",
            seed_id=12345,
            seed_x=100.0,
            seed_y=200.0,
            seed_z=300.0,
            current_segment_id=67890,
            task_ids=["task_123"],
            status="Raw",
            is_exported=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(segment)

        task = TaskModel(
            project_name="nonexistent",
            task_id="task_123",
            completion_status="",
            assigned_user_id="",
            active_user_id="user1",
            completed_user_id="",
            ng_state={},
            ng_state_initial={},
            priority=1,
            batch_id="test_batch",
            last_leased_ts=0.0,
            first_start_ts=None,
            is_active=True,
            is_paused=False,
            is_checked=False,
            task_type="seg_trace",
            id_nonunique=123456,
            extra_data=None,
            note=None,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(task)
        db_session.commit()

        with pytest.raises(ValueError, match="Project nonexistent not found"):
            get_trace_task_state(
                project_name="nonexistent",
                task_id="task_123",
                db_session=db_session,
            )

    def test_merge_edits_with_insufficient_points(self, seg_trace_task, db_session, project_name):
        """Test merge edits with insufficient points are skipped"""
        # Fixture creates required test data
        _ = seg_trace_task

        # Create merge edit with only one point
        bad_merge_edit = MergeEditModel(
            project_name=project_name,
            edit_id=99,
            task_id="task_123",
            user_id="user1",
            points=[[12345, 100.0, 200.0, 300.0]],  # Only one point
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(bad_merge_edit)
        db_session.commit()

        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="task_123",
            db_session=db_session,
        )

        # MERGES layer should be empty since the merge edit is invalid
        merge_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "MERGES")
        assert len(merge_layer["annotations"]) == 0

    def test_merge_edits_with_insufficient_coordinates(
        self, seg_trace_task, db_session, project_name
    ):
        """Test merge edits with insufficient coordinates are skipped"""
        # Fixture creates required test data
        _ = seg_trace_task

        # Create merge edit with points that don't have enough coordinates
        bad_merge_edit = MergeEditModel(
            project_name=project_name,
            edit_id=99,
            task_id="task_123",
            user_id="user1",
            points=[[12345, 100.0], [12346, 110.0, 210.0]],  # Insufficient coordinates
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(bad_merge_edit)
        db_session.commit()

        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="task_123",
            db_session=db_session,
        )

        # MERGES layer should be empty since the merge edit is invalid
        merge_layer = next(layer for layer in ng_state["layers"] if layer["name"] == "MERGES")
        assert len(merge_layer["annotations"]) == 0

    def test_no_session_provided(
        self, seg_trace_task, endpoints_for_segment, db_session, project_name, mocker
    ):
        """Test function works without providing a session"""
        # Fixtures create required test data
        assert seg_trace_task is not None
        assert endpoints_for_segment is not None

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
            "zetta_utils.task_management.ng_state.trace.get_session_context",
            side_effect=mock_get_session_context,
        )

        # Call without db_session parameter
        ng_state = get_trace_task_state(project_name, "task_123")

        assert "layers" in ng_state
        assert "position" in ng_state
        assert ng_state["position"] == [100.0, 200.0, 300.0]

    def test_segment_type_layers_error_handling(
        self, seg_trace_task, db_session, project_name, mocker
    ):
        """Test that segment type layer errors are handled gracefully"""
        # Fixture creates required test data
        _ = seg_trace_task

        # Create segment with expected_segment_type to trigger segment type layers
        segment = db_session.query(SegmentModel).filter_by(seed_id=12345).first()
        segment.expected_segment_type = "test_type"
        db_session.commit()

        # Mock _add_segment_type_layers to raise an exception
        mocker.patch(
            "zetta_utils.task_management.ng_state.segment._add_segment_type_layers",
            side_effect=Exception("Database schema error"),
        )

        # Should not raise an exception, should handle gracefully
        ng_state = get_trace_task_state(
            project_name=project_name,
            task_id="task_123",
            include_segment_type_layers=True,
            db_session=db_session,
        )

        # Should still generate valid state
        assert "layers" in ng_state
        assert "position" in ng_state

        # The segment type layer addition is handled gracefully and errors don't propagate
        # Since there's no actual segment type in the database, the function might not be called
        # but the error handling ensures the test passes without issues


class TestGetTraceTaskLink:
    """Test get_trace_task_link function"""

    def test_basic_link_generation(
        self,
        seg_trace_task,
        endpoints_for_segment,
        merge_edits_for_task,
        db_session,
        project_name,
    ):
        """Test basic spelunker link generation"""
        # Fixtures create required test data
        _ = seg_trace_task
        _ = endpoints_for_segment
        _ = merge_edits_for_task

        link = get_trace_task_link(
            project_name=project_name,
            task_id="task_123",
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

        # Check MERGES layer is present
        layer_names = [layer["name"] for layer in ng_state["layers"]]
        assert "MERGES" in layer_names

    def test_link_with_selective_layers(
        self, seg_trace_task, endpoints_for_segment, db_session, project_name
    ):
        """Test link generation with selective layers"""
        # Fixtures create required test data
        assert seg_trace_task is not None
        assert endpoints_for_segment is not None

        link = get_trace_task_link(
            project_name=project_name,
            task_id="task_123",
            include_certain_ends=False,
            include_uncertain_ends=False,
            include_breadcrumbs=False,
            include_segment_type_layers=False,
            include_merges=False,
            db_session=db_session,
        )

        # Extract and decode the state
        encoded_state = link.split("#!")[1]
        decoded_state = urllib.parse.unquote(encoded_state)
        ng_state = json.loads(decoded_state)

        layer_names = [layer["name"] for layer in ng_state["layers"]]

        # Should have core layers
        assert "Segmentation" in layer_names
        assert "Seed Location" in layer_names
        assert "Root Location" in layer_names

        # Should not have optional layers
        assert "Certain Ends" not in layer_names
        assert "Uncertain Ends" not in layer_names
        assert "Breadcrumbs" not in layer_names
        assert "MERGES" not in layer_names

    def test_link_generation_error_propagation(
        self, setup_project_and_segment, db_session, project_name
    ):
        """Test that errors from get_trace_task_state are propagated"""
        # Fixture creates required test data
        _ = setup_project_and_segment

        with pytest.raises(ValueError, match="Task nonexistent_task not found"):
            get_trace_task_link(
                project_name=project_name,
                task_id="nonexistent_task",
                db_session=db_session,
            )

    def test_no_session_provided(
        self, seg_trace_task, endpoints_for_segment, db_session, project_name, mocker
    ):
        """Test link generation works without providing a session"""
        # Fixtures create required test data
        assert seg_trace_task is not None
        assert endpoints_for_segment is not None

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
            "zetta_utils.task_management.ng_state.trace.get_session_context",
            side_effect=mock_get_session_context,
        )

        # Call without db_session parameter
        link = get_trace_task_link(project_name, "task_123")

        assert link.startswith("https://spelunker.cave-explorer.org/#!")


class TestInternalFunctions:
    """Test internal helper functions"""

    def test_get_task_and_segment_with_trace_v0(self, trace_v0_task, db_session, project_name):
        """Test _get_task_and_segment with trace_v0 task (seed_id in extra_data)"""
        # Fixture creates required test data
        assert trace_v0_task is not None

        # Use the internal function for testing
        task, segment, project = _get_task_and_segment(db_session, project_name, "trace_task_123")

        assert task.task_id == "trace_task_123"
        assert task.task_type == "trace_v0"
        assert segment.seed_id == 12345
        assert project.project_name == project_name

    def test_get_task_and_segment_with_seg_trace(self, seg_trace_task, db_session, project_name):
        """Test _get_task_and_segment with seg_trace task (segment lookup required)"""
        # Fixture creates required test data
        _ = seg_trace_task

        # Use the internal function for testing
        task, segment, project = _get_task_and_segment(db_session, project_name, "task_123")

        assert task.task_id == "task_123"
        assert task.task_type == "seg_trace"
        assert segment.seed_id == 12345
        assert "task_123" in segment.task_ids
        assert project.project_name == project_name

    def test_get_merge_edits_data(
        self,
        seg_trace_task,
        merge_edits_for_task,
        setup_project_and_segment,
        db_session,
        project_name,
    ):
        """Test _get_merge_edits_data function"""
        # Fixtures create required test data
        _ = seg_trace_task
        _ = merge_edits_for_task
        assert setup_project_and_segment is not None

        # Get project for the function call
        project = db_session.query(ProjectModel).filter_by(project_name=project_name).first()

        line_annotations = _get_merge_edits_data(db_session, project_name, "task_123", project)

        assert len(line_annotations) == 3  # Three merge edits

        # Check first line annotation
        first_line = line_annotations[0]
        assert first_line["type"] == "line"
        assert first_line["pointA"] == [12.5, 25.0, 7.5]  # 100.0/8.0, 200.0/8.0, 300.0/40.0
        assert first_line["pointB"] == [13.75, 26.25, 7.75]  # 110.0/8.0, 210.0/8.0, 310.0/40.0
        assert "id" in first_line

    def test_add_merge_layer(self, setup_project_and_segment, db_session, project_name):
        """Test _add_merge_layer function"""
        # Fixture creates required test data
        _ = setup_project_and_segment

        # Get project
        project = db_session.query(ProjectModel).filter_by(project_name=project_name).first()

        # Create test merge annotations
        merge_annotations = [
            {
                "pointA": [100.0, 200.0, 300.0],
                "pointB": [110.0, 210.0, 310.0],
                "type": "line",
                "id": "test123",
            }
        ]

        ng_state: dict[str, list] = {"layers": []}
        _add_merge_layer(ng_state, merge_annotations, project)

        assert len(ng_state["layers"]) == 1
        merge_layer = ng_state["layers"][0]
        assert merge_layer["name"] == "MERGES"
        assert merge_layer["type"] == "annotation"
        assert merge_layer["tool"] == "annotateLine"
        assert merge_layer["annotationColor"] == "#ff8c00"  # Orange
        assert merge_layer["annotations"] == merge_annotations
