"""Tests for trace_v0 task type"""

# pylint: disable=unused-argument,redefined-outer-name

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from zetta_utils.task_management.db.models import (
    EndpointModel,
    ProjectModel,
    SegmentModel,
)
from zetta_utils.task_management.db.session import get_db_session, get_session_context
from zetta_utils.task_management.task_types.trace_v0 import (
    count_annotations,
    count_visible_segments,
    create_trace_v0_task,
    extract_annotations_from_ng_state,
    generate_trace_v0_ng_state,
    get_layer_by_name,
    get_visible_segment_id,
    handle_trace_v0_completion,
    round_coords,
    verify_trace_v0,
)
from zetta_utils.task_management.types import Task


class TestHelperFunctions:
    """Test helper functions"""

    def test_get_layer_by_name_found(self):
        """Test finding a layer by name"""
        ng_state = {
            "layers": [
                {"name": "Layer1", "type": "image"},
                {"name": "Layer2", "type": "segmentation"},
                {"name": "Layer3", "type": "annotation"},
            ]
        }

        layer = get_layer_by_name(ng_state, "Layer2")
        assert layer is not None
        assert layer["name"] == "Layer2"
        assert layer["type"] == "segmentation"

    def test_get_layer_by_name_not_found(self):
        """Test when layer is not found"""
        ng_state = {
            "layers": [
                {"name": "Layer1", "type": "image"},
            ]
        }

        layer = get_layer_by_name(ng_state, "NonExistent")
        assert layer is None

    def test_get_layer_by_name_empty_state(self):
        """Test with empty state"""
        ng_state: dict[str, list] = {}
        layer = get_layer_by_name(ng_state, "Layer1")
        assert layer is None

    def test_get_visible_segment_id(self):
        """Test getting visible segment ID"""
        layer = {"segments": ["!12345", "67890", "!11111"]}

        segment_id = get_visible_segment_id(layer)
        assert segment_id == "67890"

    def test_get_visible_segment_id_all_hidden(self):
        """Test when all segments are hidden"""
        layer = {"segments": ["!12345", "!67890", "!11111"]}

        segment_id = get_visible_segment_id(layer)
        assert segment_id is None

    def test_get_visible_segment_id_empty(self):
        """Test with no segments"""
        layer: dict[str, list] = {"segments": []}
        segment_id = get_visible_segment_id(layer)
        assert segment_id is None

    def test_count_visible_segments(self):
        """Test counting visible segments"""
        layer = {"segments": ["12345", "!67890", "11111", "!22222", "33333"]}

        count = count_visible_segments(layer)
        assert count == 3

    def test_count_visible_segments_none(self):
        """Test counting with no layer"""
        count = count_visible_segments(None)  # type: ignore[arg-type]
        assert count == 0

    def test_count_annotations(self):
        """Test counting annotations"""
        layer = {
            "annotations": [
                {"point": [1, 2, 3]},
                {"point": [4, 5, 6]},
                {"point": [7, 8, 9]},
            ]
        }

        count = count_annotations(layer)
        assert count == 3

    def test_count_annotations_empty(self):
        """Test counting with empty annotations"""
        layer: dict[str, list] = {"annotations": []}
        count = count_annotations(layer)
        assert count == 0

    def test_count_annotations_none(self):
        """Test counting with no layer"""
        count = count_annotations(None)  # type: ignore[arg-type]
        assert count == 0

    def test_round_coords(self):
        """Test coordinate rounding"""
        x, y, z = round_coords(1.23456789, 2.34567891, 3.45678912)
        assert x == 1.234568
        assert y == 2.345679
        assert z == 3.456789

    def test_extract_annotations_from_ng_state(self):
        """Test extracting all annotations from neuroglancer state"""
        ng_state = {
            "layers": [
                {
                    "name": "Seed Location",
                    "type": "annotation",
                    "annotations": [{"point": [100, 200, 300]}],
                },
                {
                    "name": "Root Location",
                    "type": "annotation",
                    "annotations": [{"point": [150, 250, 350]}],
                },
                {
                    "name": "Certain Ends",
                    "type": "annotation",
                    "annotations": [{"point": [110, 210, 310]}, {"point": [120, 220, 320]}],
                },
                {
                    "name": "Uncertain Ends",
                    "type": "annotation",
                    "annotations": [{"point": [130, 230, 330]}],
                },
                {
                    "name": "Breadcrumbs",
                    "type": "annotation",
                    "annotations": [{"point": [140, 240, 340]}, {"point": [160, 260, 360]}],
                },
            ]
        }

        annotations = extract_annotations_from_ng_state(ng_state)

        assert annotations["seed"] == [100, 200, 300]
        assert annotations["root"] == [150, 250, 350]
        assert len(annotations["certain_ends"]) == 2
        assert annotations["certain_ends"][0] == [110, 210, 310]
        assert len(annotations["uncertain_ends"]) == 1
        assert annotations["uncertain_ends"][0] == [130, 230, 330]
        assert len(annotations["breadcrumbs"]) == 2

    def test_extract_annotations_empty_state(self):
        """Test extracting from empty state"""
        ng_state: dict[str, list] = {"layers": []}
        annotations = extract_annotations_from_ng_state(ng_state)

        assert annotations["seed"] is None
        assert annotations["root"] is None
        assert not annotations["certain_ends"]
        assert not annotations["uncertain_ends"]
        assert not annotations["breadcrumbs"]


class TestVerifyTraceV0:
    """Test trace_v0 verification"""

    def test_verify_faulty_task(self, sample_task):
        """Test verification skips faulty tasks"""
        result = verify_trace_v0("test_project", sample_task, "Faulty Task")
        assert result.passed is True
        assert "faulty" in result.message.lower()

    def test_verify_missing_layers(self, sample_task):
        """Test verification fails with missing layers"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation"},
            # Missing other required layers
        ]

        result = verify_trace_v0("test_project", sample_task, "Done")
        assert result.passed is False
        assert "Missing required layers" in result.message

    def test_verify_wrong_segment_count(self, sample_task):
        """Test verification fails with wrong segment count"""
        # Add all required layers
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1", "2", "3"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
        ]

        result = verify_trace_v0("test_project", sample_task, "Done")
        assert result.passed is False
        assert "exactly 1 visible segment" in result.message

    def test_verify_done_without_root(self, sample_task):
        """Test Done status requires root location"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
            # Missing Root Location
        ]

        result = verify_trace_v0("test_project", sample_task, "Done")
        assert result.passed is False
        assert "requires Root Location" in result.message

    def test_verify_done_insufficient_certain_ends(self, sample_task):
        """Test Done status requires sufficient certain ends"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": [{"point": [1, 2, 3]}]},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
            {"name": "Root Location", "type": "annotation", "annotations": [{"point": [4, 5, 6]}]},
        ]

        result = verify_trace_v0("test_project", sample_task, "Done")
        assert result.passed is False
        assert "Insufficient certain end count" in result.message

    def test_verify_done_success(self, sample_task):
        """Test successful Done verification"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {
                "name": "Certain Ends",
                "type": "annotation",
                "annotations": [
                    {"point": [1, 2, 3]},
                    {"point": [4, 5, 6]},
                    {"point": [7, 8, 9]},
                ],
            },
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
            {
                "name": "Root Location",
                "type": "annotation",
                "annotations": [{"point": [10, 11, 12]}],
            },
        ]

        result = verify_trace_v0("test_project", sample_task, "Done")
        assert result.passed is True
        assert "3 certain ends" in result.message

    def test_verify_cant_continue_no_uncertain(self, sample_task):
        """Test Can't Continue requires uncertain ends"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},  # Empty
            {"name": "Seed Location", "type": "annotation", "annotations": []},
        ]

        result = verify_trace_v0("test_project", sample_task, "Can't Continue")
        assert result.passed is False
        assert "at least 1 uncertain end" in result.message

    def test_verify_cant_continue_success(self, sample_task):
        """Test successful Can't Continue verification"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {
                "name": "Uncertain Ends",
                "type": "annotation",
                "annotations": [
                    {"point": [1, 2, 3]},
                    {"point": [4, 5, 6]},
                ],
            },
            {"name": "Seed Location", "type": "annotation", "annotations": []},
        ]

        result = verify_trace_v0("test_project", sample_task, "Can't Continue")
        assert result.passed is True
        assert "2 uncertain ends" in result.message

    def test_verify_merger_success(self, sample_task):
        """Test Merger status verification"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
        ]

        result = verify_trace_v0("test_project", sample_task, "Merger")
        assert result.passed is True
        assert "Merger" in result.message

    def test_verify_wrong_cell_type_success(self, sample_task):
        """Test Wrong Cell Type status verification"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
        ]

        result = verify_trace_v0("test_project", sample_task, "Wrong Cell Type")
        assert result.passed is True
        assert "Wrong Cell Type" in result.message

    def test_verify_invalid_status(self, sample_task):
        """Test invalid completion status"""
        sample_task["ng_state"]["layers"] = [
            {"name": "Segmentation", "type": "segmentation", "segments": ["1"]},
            {"name": "Breadcrumbs", "type": "annotation", "annotations": []},
            {"name": "Certain Ends", "type": "annotation", "annotations": []},
            {"name": "Uncertain Ends", "type": "annotation", "annotations": []},
            {"name": "Seed Location", "type": "annotation", "annotations": []},
        ]

        result = verify_trace_v0("test_project", sample_task, "Invalid Status")
        assert result.passed is False
        assert "Invalid completion status" in result.message


class TestHandleTraceV0Completion:
    """Test trace_v0 completion handler"""

    def test_handle_faulty_task(self, sample_task):
        """Test handler skips faulty tasks"""
        # Should not raise any errors
        handle_trace_v0_completion("test_project", sample_task, "Faulty Task")

    def test_handle_missing_extra_data(self, sample_task):
        """Test handler with missing extra_data"""
        sample_task["extra_data"] = None

        # Should log error but not raise
        handle_trace_v0_completion("test_project", sample_task, "Done")

    def test_handle_missing_seed_id(self, sample_task):
        """Test handler with missing seed_id"""
        sample_task["extra_data"] = {}

        # Should log error but not raise
        handle_trace_v0_completion("test_project", sample_task, "Done")

    def test_handle_completion_done(
        self, clean_db, db_session, existing_project, existing_segment, sample_task
    ):
        """Test handling Done completion"""
        # Setup task with proper ng_state
        sample_task["task_id"] = "test_task_123"
        sample_task["extra_data"] = {"seed_id": existing_segment.seed_id}
        sample_task["completed_user_id"] = "user123"
        sample_task["ng_state"] = {
            "layers": [
                {"name": "Segmentation", "type": "segmentation", "segments": ["98765"]},
                {
                    "name": "Root Location",
                    "type": "annotation",
                    "annotations": [{"point": [100, 200, 300.5]}],
                },
                {
                    "name": "Certain Ends",
                    "type": "annotation",
                    "annotations": [{"point": [110, 210, 310]}, {"point": [120, 220, 320]}],
                },
                {
                    "name": "Uncertain Ends",
                    "type": "annotation",
                    "annotations": [{"point": [130, 230, 330]}],
                },
                {
                    "name": "Breadcrumbs",
                    "type": "annotation",
                    "annotations": [{"point": [140, 240, 340]}],
                },
            ]
        }

        # Run completion handler
        handle_trace_v0_completion(existing_project, sample_task, "Done")

        # Need to use a fresh session to see changes made by the completion handler
        with get_session_context() as fresh_session:
            segment = (
                fresh_session.query(SegmentModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .first()
            )

            assert segment.current_segment_id == 98765
            assert segment.status == "Completed"
            assert segment.root_x == 100
            assert segment.root_y == 200
            assert segment.root_z == 300  # Should be 300.5 - 0.5
            assert "test_task_123" in segment.task_ids

            # Check endpoints were created
            endpoints = (
                fresh_session.query(EndpointModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .all()
            )

        assert len(endpoints) == 4  # 2 certain, 1 uncertain, 1 breadcrumb

        certain_count = sum(1 for e in endpoints if e.status == "CERTAIN")
        uncertain_count = sum(1 for e in endpoints if e.status == "UNCERTAIN")
        breadcrumb_count = sum(1 for e in endpoints if e.status == "BREADCRUMB")

        assert certain_count == 2
        assert uncertain_count == 1
        assert breadcrumb_count == 1

        # Check user assignment
        for endpoint in endpoints:
            assert endpoint.user == "user123"

    def test_handle_completion_merger(
        self, clean_db, db_session, existing_project, existing_segment, sample_task
    ):
        """Test handling Merger completion"""
        sample_task["task_id"] = "test_task_123"
        sample_task["extra_data"] = {"seed_id": existing_segment.seed_id}
        sample_task["ng_state"] = {
            "layers": [
                {"name": "Segmentation", "type": "segmentation", "segments": ["98765"]},
            ]
        }

        handle_trace_v0_completion(existing_project, sample_task, "Merger")

        # Need to use a fresh session to see changes made by the completion handler
        with get_session_context() as fresh_session:
            segment = (
                fresh_session.query(SegmentModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .first()
            )

            assert segment.status == "Abandoned"

    def test_handle_completion_cant_continue(
        self, clean_db, db_session, existing_project, existing_segment, sample_task
    ):
        """Test handling Can't Continue completion"""
        sample_task["task_id"] = "test_task_123"
        sample_task["extra_data"] = {"seed_id": existing_segment.seed_id}
        sample_task["ng_state"] = {
            "layers": [
                {"name": "Segmentation", "type": "segmentation", "segments": ["98765"]},
            ]
        }

        handle_trace_v0_completion(existing_project, sample_task, "Can't Continue")

        # Need to use a fresh session to see changes made by the completion handler
        with get_session_context() as fresh_session:
            segment = (
                fresh_session.query(SegmentModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .first()
            )

            assert segment.status == "WIP"

    def test_handle_completion_replaces_endpoints(
        self, clean_db, db_session, existing_project, existing_segment, sample_task
    ):
        """Test that completion replaces existing endpoints"""
        # Create some existing endpoints
        for i in range(3):
            endpoint = EndpointModel(
                project_name=existing_project,
                seed_id=existing_segment.seed_id,
                x=i * 10,
                y=i * 20,
                z=i * 30,
                status="CERTAIN",
                user="old_user",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db_session.add(endpoint)
        db_session.commit()

        # Verify old endpoints exist
        old_endpoints = (
            db_session.query(EndpointModel)
            .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
            .all()
        )
        assert len(old_endpoints) == 3

        # Setup task with new endpoints
        sample_task["task_id"] = "test_task_123"
        sample_task["extra_data"] = {"seed_id": existing_segment.seed_id}
        sample_task["completed_user_id"] = "new_user"
        sample_task["ng_state"] = {
            "layers": [
                {"name": "Segmentation", "type": "segmentation", "segments": ["98765"]},
                {
                    "name": "Certain Ends",
                    "type": "annotation",
                    "annotations": [{"point": [1000, 2000, 3000]}],
                },
            ]
        }

        handle_trace_v0_completion(existing_project, sample_task, "Done")

        # Need to use a fresh session to see changes made by the completion handler
        with get_session_context() as fresh_session:
            # Check old endpoints were deleted and new ones created
            new_endpoints = (
                fresh_session.query(EndpointModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .all()
            )

            assert len(new_endpoints) == 1
            assert new_endpoints[0].x == 1000
            assert new_endpoints[0].user == "new_user"

    def test_handle_completion_segment_not_found(
        self, clean_db, db_session, existing_project, sample_task
    ):
        """Test handling when segment not found"""
        # Make sure to use a seed_id that definitely doesn't exist
        non_existent_seed_id = 9999999
        sample_task["extra_data"] = {"seed_id": non_existent_seed_id}
        sample_task["ng_state"] = {
            "layers": [
                {"name": "Segmentation", "type": "segmentation", "segments": ["98765"]},
            ]
        }

        # This should raise ValueError when segment is not found
        with pytest.raises(ValueError, match=f"Segment {non_existent_seed_id} not found"):
            handle_trace_v0_completion(existing_project, sample_task, "Done")


class TestGenerateTraceV0NgState:
    """Test neuroglancer state generation"""

    def test_generate_ng_state_basic(
        self, clean_db, db_session, existing_project, existing_segment
    ):
        """Test basic neuroglancer state generation"""
        ng_state = generate_trace_v0_ng_state(existing_project, existing_segment)

        # Check basic structure
        assert "dimensions" in ng_state
        assert "position" in ng_state
        assert "layers" in ng_state

        # Check position matches seed
        assert ng_state["position"] == [
            existing_segment.seed_x,
            existing_segment.seed_y,
            existing_segment.seed_z,
        ]

        # Check required layers exist
        layer_names = [layer["name"] for layer in ng_state["layers"]]
        assert "Image" in layer_names
        assert "Segmentation" in layer_names
        assert "Seed Location" in layer_names
        assert "Root Location" in layer_names
        assert "Certain Ends" in layer_names
        assert "Uncertain Ends" in layer_names
        assert "Breadcrumbs" in layer_names

        # Check segmentation layer has correct segment
        seg_layer = next(l for l in ng_state["layers"] if l["name"] == "Segmentation")
        assert seg_layer["segments"] == [str(existing_segment.current_segment_id)]

        # Check seed location annotation
        seed_layer = next(l for l in ng_state["layers"] if l["name"] == "Seed Location")
        assert len(seed_layer["annotations"]) == 1
        assert seed_layer["annotations"][0]["point"] == [
            existing_segment.seed_x,
            existing_segment.seed_y,
            existing_segment.seed_z,
        ]

    def test_generate_ng_state_with_endpoints(
        self, clean_db, db_session, existing_project, existing_segment
    ):
        """Test state generation with existing endpoints"""
        # Create some endpoints
        endpoints_data = [
            (100, 200, 300, "CERTAIN"),
            (110, 210, 310, "CERTAIN"),
            (120, 220, 320, "UNCERTAIN"),
            (130, 230, 330, "BREADCRUMB"),
        ]

        for x, y, z, status in endpoints_data:
            endpoint = EndpointModel(
                project_name=existing_project,
                seed_id=existing_segment.seed_id,
                x=x,
                y=y,
                z=z,
                status=status,
                user="test_user",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db_session.add(endpoint)
        db_session.commit()

        # Generate state
        ng_state = generate_trace_v0_ng_state(existing_project, existing_segment)

        # Check endpoints are included
        certain_layer = next(l for l in ng_state["layers"] if l["name"] == "Certain Ends")
        assert len(certain_layer["annotations"]) == 2

        uncertain_layer = next(l for l in ng_state["layers"] if l["name"] == "Uncertain Ends")
        assert len(uncertain_layer["annotations"]) == 1

        breadcrumb_layer = next(l for l in ng_state["layers"] if l["name"] == "Breadcrumbs")
        assert len(breadcrumb_layer["annotations"]) == 1

        # Check annotation format
        for annotation in certain_layer["annotations"]:
            assert "point" in annotation
            assert "type" in annotation
            assert "id" in annotation
            assert annotation["type"] == "point"

    def test_generate_ng_state_no_current_segment(self, clean_db, db_session, existing_project):
        """Test state generation when segment has no current_segment_id"""
        # Create segment without current_segment_id
        segment = SegmentModel(
            project_name=existing_project,
            seed_id=99999,
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

        ng_state = generate_trace_v0_ng_state(existing_project, segment)

        seg_layer = next(l for l in ng_state["layers"] if l["name"] == "Segmentation")
        assert seg_layer["segments"] == []

    def test_generate_ng_state_with_extra_layers(self, clean_db, db_session, project_name):
        """Test state generation with project extra layers"""
        # Create project with extra layers
        project = ProjectModel(
            project_name=project_name,
            segmentation_path="precomputed://test",
            sv_resolution_x=8.0,
            sv_resolution_y=8.0,
            sv_resolution_z=40.0,
            extra_layers=[
                {"name": "Extra Layer 1", "type": "image", "source": "precomputed://extra1"},
                {
                    "name": "Extra Layer 2",
                    "type": "segmentation",
                    "source": "precomputed://extra2",
                },
            ],
        )
        db_session.add(project)

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

        ng_state = generate_trace_v0_ng_state(project_name, segment)

        # Check extra layers are appended
        layer_names = [layer["name"] for layer in ng_state["layers"]]
        assert "Extra Layer 1" in layer_names
        assert "Extra Layer 2" in layer_names

    def test_generate_ng_state_project_not_found(self, clean_db, db_session, existing_segment):
        """Test error when project not found"""
        with pytest.raises(ValueError, match="Project nonexistent not found"):
            generate_trace_v0_ng_state("nonexistent", existing_segment)


class TestCreateTraceV0Task:
    """Test trace_v0 task creation"""

    @patch("zetta_utils.task_management.task_types.trace_v0.create_task")
    @patch("zetta_utils.task_management.task_types.trace_v0.generate_trace_v0_ng_state")
    def test_create_task_basic(
        self,
        mock_generate_ng,
        mock_create_task,
        clean_db,
        db_session,
        existing_project,
        existing_segment,
    ):
        """Test basic task creation"""
        mock_ng_state = {"test": "state"}
        mock_generate_ng.return_value = mock_ng_state

        # Mock create_task to return the task_id that was passed to it
        def mock_create_task_impl(**kwargs):
            return kwargs["data"]["task_id"]

        mock_create_task.side_effect = mock_create_task_impl

        task_id = create_trace_v0_task(existing_project, existing_segment, {})

        # Check ng state generation
        mock_generate_ng.assert_called_once_with(existing_project, existing_segment)

        # Check task creation
        mock_create_task.assert_called_once()
        call_args = mock_create_task.call_args[1]
        task_data = call_args["data"]

        assert task_data["task_type"] == "trace_v0"
        assert task_data["ng_state"] == mock_ng_state
        assert task_data["ng_state_initial"] == mock_ng_state  # Should be a copy
        assert task_data["extra_data"]["seed_id"] == existing_segment.seed_id
        assert task_data["priority"] == 50  # Default
        assert task_data["batch_id"] == "default"
        assert task_data["is_active"] is True
        assert task_data["is_paused"] is False

        # Check that the returned task_id matches what was generated
        assert task_id.startswith(f"trace_{existing_segment.seed_id}_")

        # Check segment was updated - need to use fresh session
        with get_session_context() as fresh_session:
            segment = (
                fresh_session.query(SegmentModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .first()
            )

            # The actual task_id that was generated should be in the segment's task_ids
            assert task_id in segment.task_ids

    @patch("zetta_utils.task_management.task_types.trace_v0.create_task")
    @patch("zetta_utils.task_management.task_types.trace_v0.generate_trace_v0_ng_state")
    def test_create_task_with_kwargs(
        self,
        mock_generate_ng,
        mock_create_task,
        clean_db,
        db_session,
        existing_project,
        existing_segment,
    ):
        """Test task creation with custom kwargs"""
        mock_generate_ng.return_value = {"test": "state"}

        # Mock create_task to return the task_id that was passed to it
        def mock_create_task_impl(**kwargs):
            return kwargs["data"]["task_id"]

        mock_create_task.side_effect = mock_create_task_impl

        kwargs = {
            "priority": 100,
            "batch_id": "custom_batch",
            "is_paused": True,
        }

        create_trace_v0_task(existing_project, existing_segment, kwargs)

        # Check custom values were used
        call_args = mock_create_task.call_args[1]
        task_data = call_args["data"]

        assert task_data["priority"] == 100
        assert task_data["batch_id"] == "custom_batch"
        assert task_data["is_paused"] is True

    @patch("zetta_utils.task_management.task_types.trace_v0.create_task")
    @patch("zetta_utils.task_management.task_types.trace_v0.generate_trace_v0_ng_state")
    def test_create_task_idempotent(
        self,
        mock_generate_ng,
        mock_create_task,
        clean_db,
        db_session,
        existing_project,
        existing_segment,
    ):
        """Test task creation doesn't duplicate task_ids"""
        mock_generate_ng.return_value = {"test": "state"}

        # Pre-add a task_id to segment
        existing_task_id = f"trace_{existing_segment.seed_id}_12345"
        existing_segment.task_ids = [existing_task_id]
        db_session.commit()

        # Mock create_task to return the task_id that was passed to it
        def mock_create_task_impl(**kwargs):
            return kwargs["data"]["task_id"]

        mock_create_task.side_effect = mock_create_task_impl

        created_id = create_trace_v0_task(existing_project, existing_segment, {})

        # Check task_id wasn't duplicated - need to use fresh session
        with get_session_context() as fresh_session:
            segment = (
                fresh_session.query(SegmentModel)
                .filter_by(project_name=existing_project, seed_id=existing_segment.seed_id)
                .first()
            )

            # Should have the existing task_id and the new one
            assert len(segment.task_ids) == 2
            assert existing_task_id in segment.task_ids
            assert created_id in segment.task_ids
            assert created_id != existing_task_id  # Should be different


@pytest.fixture
def sample_task():
    """Create a sample task for testing"""
    return Task(
        task_id="test_task_123",
        task_type="trace_v0",
        ng_state={"layers": []},
        ng_state_initial={},
        completion_status="",
        assigned_user_id="",
        active_user_id="user123",
        completed_user_id="",
        priority=50,
        batch_id="default",
        last_leased_ts=0.0,
        is_active=True,
        is_paused=False,
        is_checked=False,
        extra_data={"seed_id": 12345},
    )


@pytest.fixture(autouse=True)
def patch_get_db_session(postgres_container, monkeypatch):
    """Patch get_db_session to use test database for all sessions"""
    connection_url = postgres_container.get_connection_url()

    def mock_get_db_session(engine_url=None):
        # Always use the test database URL
        return get_db_session(engine_url=connection_url)

    monkeypatch.setattr(
        "zetta_utils.task_management.db.session.get_db_session", mock_get_db_session
    )


@pytest.fixture
def existing_project(clean_db, db_session, project_name):
    """Create a project for testing"""
    project = ProjectModel(
        project_name=project_name,
        segmentation_path="precomputed://gs://test-bucket/segmentation",
        sv_resolution_x=8.0,
        sv_resolution_y=8.0,
        sv_resolution_z=40.0,
    )
    db_session.add(project)
    db_session.commit()
    # Need to refresh to ensure the project is queryable
    db_session.refresh(project)
    return project_name


@pytest.fixture
def existing_segment(existing_project, db_session):
    """Create a segment for testing"""
    segment = SegmentModel(
        project_name=existing_project,
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
    return segment
