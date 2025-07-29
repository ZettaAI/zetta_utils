"""
Generate neuroglancer links for trace tasks with merge annotations.

This module provides functions to create neuroglancer states and links
for trace tasks, including all segment data, endpoints, and merge edits
as line annotations.
"""

import json
import urllib.parse
import uuid
from typing import Any, Dict, List

from zetta_utils.task_management.db.models import (
    MergeEditModel,
    ProjectModel,
    SegmentModel,
    TaskModel,
)
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment_link import (
    _add_core_layers,
    _add_endpoint_layers,
    _add_segment_type_layers,
    _build_base_ng_state,
    _get_endpoints_data,
    _get_segment_and_project,
)


def _get_task_and_segment(session, project_name: str, task_id: str):
    """Get task and associated segment from database."""
    task = (
        session.query(TaskModel)
        .filter_by(project_name=project_name, task_id=task_id)
        .first()
    )
    if not task:
        raise ValueError(f"Task {task_id} not found in project {project_name}")

    # Try to get seed_id from task's extra_data first (for trace_v0 tasks)
    extra_data = task.extra_data or {}
    seed_id = extra_data.get("seed_id")

    if seed_id:
        # Found seed_id in extra_data
        segment, project = _get_segment_and_project(session, project_name, seed_id)
        return task, segment, project

    # If no seed_id in extra_data, look for segment that contains this task_id
    # This handles other task types like 'seg_trace'
    segment = (
        session.query(SegmentModel)
        .filter_by(project_name=project_name)
        .filter(SegmentModel.task_ids.op("&&")([task_id]))
        .first()
    )

    if not segment:
        raise ValueError(f"Task {task_id} is not associated with any segment")

    # Get project info directly
    project = (
        session.query(ProjectModel)
        .filter_by(project_name=project_name)
        .first()
    )

    if not project:
        raise ValueError(f"Project {project_name} not found")

    return task, segment, project


def _get_merge_edits_data(  # pylint: disable=unused-argument
    session, project_name: str, task_id: str, project
) -> List[Dict[str, Any]]:
    """Get merge edits for the task and convert to line annotations."""
    merge_edits = (
        session.query(MergeEditModel)
        .filter_by(project_name=project_name, task_id=task_id)
        .order_by(MergeEditModel.created_at.asc())
        .all()
    )

    line_annotations = []
    for merge_edit in merge_edits:
        if len(merge_edit.points) >= 2:
            # Each merge edit has points in format [[segment_id, x, y, z], [segment_id, x, y, z]]
            point1 = merge_edit.points[0]
            point2 = merge_edit.points[1]

            # Extract coordinates (skip segment_id)
            if len(point1) >= 4 and len(point2) >= 4:
                coord1 = [point1[1], point1[2], point1[3]]  # [x, y, z]
                coord2 = [point2[1], point2[2], point2[3]]  # [x, y, z]

                line_annotation = {
                    "pointA": coord1,
                    "pointB": coord2,
                    "type": "line",
                    "id": str(uuid.uuid4()).replace("-", ""),
                }
                line_annotations.append(line_annotation)

    return line_annotations


def _add_merge_layer(ng_state: Dict[str, Any], merge_annotations: List[Dict[str, Any]], project):
    """Add MERGES annotation layer with line annotations."""
    sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

    ng_state["layers"].append(
        {
            "type": "annotation",
            "source": {
                "url": "local://annotations",
                "transform": {
                    "outputDimensions": {
                        "x": [sv_resolution[0] * 1e-9, "m"],
                        "y": [sv_resolution[1] * 1e-9, "m"],
                        "z": [sv_resolution[2] * 1e-9, "m"],
                    }
                },
            },
            "tool": "annotateLine",
            "tab": "annotations",
            "annotationColor": "#ff8c00",  # Orange
            "annotations": merge_annotations,
            "name": "MERGES",
        }
    )


def get_trace_task_state(
    project_name: str,
    task_id: str,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
    include_segment_type_layers: bool = True,
    include_merges: bool = True,
    db_session=None,
) -> Dict[str, Any]:
    """
    Generate a neuroglancer state for a trace task with merge annotations.

    Args:
        project_name: Name of the project
        task_id: ID of the trace task
        include_certain_ends: Whether to include certain endpoints layer. Defaults to True.
        include_uncertain_ends: Whether to include uncertain endpoints layer. Defaults to True.
        include_breadcrumbs: Whether to include breadcrumbs layer. Defaults to True.
        include_segment_type_layers: Whether to include segment type layers. Defaults to True.
        include_merges: Whether to include merge edits as line annotations. Defaults to True.
        db_session: Optional database session to use. If not provided, creates a new one.

    Returns:
        Neuroglancer state dict with annotation layers including merges
    """

    def _generate_state(session):
        # Get task, segment, and project
        _, segment, project = _get_task_and_segment(session, project_name, task_id)

        # Get endpoints data
        endpoints_data = _get_endpoints_data(
            session,
            project_name,
            segment.seed_id,
            include_certain_ends,
            include_uncertain_ends,
            include_breadcrumbs,
        )

        # Build base neuroglancer state
        ng_state = _build_base_ng_state(segment, project)

        # Add core layers (extra, segmentation, seed, root)
        _add_core_layers(ng_state, segment, project)

        # Add endpoint layers
        _add_endpoint_layers(
            ng_state,
            endpoints_data,
            project,
            include_certain_ends,
            include_uncertain_ends,
            include_breadcrumbs,
        )

        # Add segment type layers (with error handling for schema issues)
        if include_segment_type_layers:
            try:
                _add_segment_type_layers(
                    ng_state, segment, project_name, session, include_segment_type_layers
                )
            except Exception:  # pylint: disable=broad-exception-caught
                # Rollback the session if there was an error and silently skip segment type layers
                session.rollback()

        # Add merge edits layer
        if include_merges:
            merge_annotations = _get_merge_edits_data(session, project_name, task_id, project)
            _add_merge_layer(ng_state, merge_annotations, project)

        return ng_state

    # Use provided session or create a new one
    if db_session is not None:
        return _generate_state(db_session)
    else:
        with get_session_context() as session:
            return _generate_state(session)


def get_trace_task_link(
    project_name: str,
    task_id: str,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
    include_segment_type_layers: bool = True,
    include_merges: bool = True,
    db_session=None,
) -> str:
    """
    Generate a spelunker link for a trace task with merge annotations.

    Args:
        project_name: Name of the project
        task_id: ID of the trace task
        include_certain_ends: Whether to include certain endpoints layer. Defaults to True.
        include_uncertain_ends: Whether to include uncertain endpoints layer. Defaults to True.
        include_breadcrumbs: Whether to include breadcrumbs layer. Defaults to True.
        include_segment_type_layers: Whether to include segment type layers. Defaults to True.
        include_merges: Whether to include merge edits as line annotations. Defaults to True.
        db_session: Optional database session to use. If not provided, creates a new one.

    Returns:
        Spelunker URL string
    """
    ng_state = get_trace_task_state(
        project_name,
        task_id,
        include_certain_ends=include_certain_ends,
        include_uncertain_ends=include_uncertain_ends,
        include_breadcrumbs=include_breadcrumbs,
        include_segment_type_layers=include_segment_type_layers,
        include_merges=include_merges,
        db_session=db_session,
    )
    encoded_state = urllib.parse.quote(json.dumps(ng_state), safe="")
    return f"https://spelunker.cave-explorer.org/#!{encoded_state}"
