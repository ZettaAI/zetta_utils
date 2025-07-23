"""
Generate neuroglancer links for segments with endpoints.

This module provides functions to create neuroglancer states and links
for segments in the database, including all their endpoints with proper
color coding.
"""

import json
import urllib.parse
import uuid

from zetta_utils.task_management.db.models import (
    EndpointModel,
    ProjectModel,
    SegmentModel,
)
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.seg_trace_utils.segment_type import (
    get_segment_type_layers,
)


def _get_segment_and_project(session, project_name: str, seed_id: int):
    """Get segment and project from database."""
    segment = (
        session.query(SegmentModel).filter_by(project_name=project_name, seed_id=seed_id).first()
    )
    if not segment:
        raise ValueError(f"Segment {seed_id} not found in project {project_name}")

    project = session.query(ProjectModel).filter_by(project_name=project_name).first()
    if not project:
        raise ValueError(f"Project {project_name} not found")

    return segment, project


def _get_endpoints_data(
    session,
    project_name: str,
    seed_id: int,
    include_certain_ends: bool,
    include_uncertain_ends: bool,
    include_breadcrumbs: bool,
):
    """Get and group endpoints by status."""
    if not (include_certain_ends or include_uncertain_ends or include_breadcrumbs):
        return {"certain_ends": [], "uncertain_ends": [], "breadcrumbs": []}

    endpoints = (
        session.query(EndpointModel).filter_by(project_name=project_name, seed_id=seed_id).all()
    )

    certain_ends = []
    uncertain_ends = []
    breadcrumbs = []

    for endpoint in endpoints:
        annotation = {
            "point": [endpoint.x, endpoint.y, endpoint.z],
            "type": "point",
            "id": str(uuid.uuid4()).replace("-", ""),
        }

        if endpoint.status == "CERTAIN":
            certain_ends.append(annotation)
        elif endpoint.status == "UNCERTAIN":
            uncertain_ends.append(annotation)
        elif endpoint.status == "BREADCRUMB":
            breadcrumbs.append(annotation)

    return {
        "certain_ends": certain_ends,
        "uncertain_ends": uncertain_ends,
        "breadcrumbs": breadcrumbs,
    }


def _build_base_ng_state(segment, project):
    """Build base neuroglancer state."""
    sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

    return {
        "dimensions": {
            "x": [sv_resolution[0] * 1e-9, "m"],
            "y": [sv_resolution[1] * 1e-9, "m"],
            "z": [sv_resolution[2] * 1e-9, "m"],
        },
        "position": [segment.seed_x, segment.seed_y, segment.seed_z],
        "crossSectionScale": 0.5,
        "projectionOrientation": [0.5, -0.5, 0.5, 0.5],
        "projectionScale": 2740.449487163767,
        "projectionDepth": 541651.3969244945,
        "layers": [],
        "showSlices": False,
        "selectedLayer": {"visible": True, "layer": "Segmentation"},
        "layout": "xy-3d",
    }


def _add_core_layers(ng_state: dict, segment, project):
    """Add core layers (extra, segmentation, seed, root)."""
    # Add extra layers first if configured in project
    if project.extra_layers:
        for layer in project.extra_layers:
            ng_state["layers"].append(layer)

    # Add segmentation layer
    ng_state["layers"].append(
        {
            "type": "segmentation",
            "source": project.segmentation_path,
            "tab": "segments",
            "segments": ([str(segment.current_segment_id)] if segment.current_segment_id else []),
            "name": "Segmentation",
        }
    )

    sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

    # Add seed location annotation (purple)
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
            "tool": "annotatePoint",
            "tab": "annotations",
            "annotationColor": "#ff00ff",  # Purple
            "annotations": [
                {
                    "point": [segment.seed_x, segment.seed_y, segment.seed_z],
                    "type": "point",
                    "id": str(uuid.uuid4()).replace("-", ""),
                }
            ],
            "name": "Seed Location",
        }
    )

    # Add root location annotation (green)
    root_annotations = []
    if all(coord is not None for coord in [segment.root_x, segment.root_y, segment.root_z]):
        root_annotations.append(
            {
                "point": [segment.root_x, segment.root_y, segment.root_z],
                "type": "point",
                "id": str(uuid.uuid4()).replace("-", ""),
            }
        )

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
            "tool": "annotatePoint",
            "tab": "annotations",
            "annotationColor": "#00ff00",  # Green
            "annotations": root_annotations,
            "name": "Root Location",
        }
    )


def _add_endpoint_layers(
    ng_state: dict,
    endpoints_data: dict,
    project,
    include_certain_ends: bool,
    include_uncertain_ends: bool,
    include_breadcrumbs: bool,
):
    """Add endpoint layers based on flags."""
    sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

    endpoint_configs = [
        (
            include_certain_ends,
            endpoints_data["certain_ends"],
            "#ffff00",
            "Certain Ends",
            "annotations",
        ),
        (
            include_uncertain_ends,
            endpoints_data["uncertain_ends"],
            "#ff0000",
            "Uncertain Ends",
            "annotations",
        ),
        (
            include_breadcrumbs,
            endpoints_data["breadcrumbs"],
            "#0400ff",
            "Breadcrumbs",
            "rendering",
        ),
    ]

    for include_flag, annotations, color, name, tab in endpoint_configs:
        if include_flag:
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
                    "tool": "annotatePoint",
                    "tab": tab,
                    "annotationColor": color,
                    "annotations": annotations,
                    "name": name,
                }
            )


def _add_segment_type_layers(
    ng_state: dict, segment, project_name: str, session, include_segment_type_layers: bool
):
    """Add segment type layers if requested."""
    if include_segment_type_layers and segment.expected_segment_type:
        segment_type_layers = get_segment_type_layers(
            project_name=project_name,
            segment_type_name=segment.expected_segment_type,
            include_names=False,
            db_session=session,
        )
        for layer in segment_type_layers:
            ng_state["layers"].append(layer)


def get_segment_ng_state(
    project_name: str,
    seed_id: int,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
    include_segment_type_layers: bool = True,
    db_session=None,
) -> dict:
    """
    Generate a neuroglancer state for a segment with all its endpoints.

    Args:
        project_name: Name of the project
        seed_id: Seed supervoxel ID (primary key)
        include_certain_ends: Whether to include certain endpoints layer. Defaults to True.
        include_uncertain_ends: Whether to include uncertain endpoints layer. Defaults to True.
        include_breadcrumbs: Whether to include breadcrumbs layer. Defaults to True.
        include_segment_type_layers: Whether to include segment type layers
            (mesh, mask, examples). Defaults to True.
        db_session: Optional database session to use. If not provided, creates a new one.

    Returns:
        Neuroglancer state dict with annotation layers for endpoints
    """

    def _generate_state(session):
        segment, project = _get_segment_and_project(session, project_name, seed_id)
        endpoints_data = _get_endpoints_data(
            session,
            project_name,
            seed_id,
            include_certain_ends,
            include_uncertain_ends,
            include_breadcrumbs,
        )
        ng_state = _build_base_ng_state(segment, project)
        _add_core_layers(ng_state, segment, project)
        _add_endpoint_layers(
            ng_state,
            endpoints_data,
            project,
            include_certain_ends,
            include_uncertain_ends,
            include_breadcrumbs,
        )
        _add_segment_type_layers(
            ng_state, segment, project_name, session, include_segment_type_layers
        )
        return ng_state

    # Use provided session or create a new one
    if db_session is not None:
        return _generate_state(db_session)
    else:
        with get_session_context() as session:
            return _generate_state(session)


def get_segment_link(
    project_name: str,
    seed_id: int,
    include_certain_ends: bool = True,
    include_uncertain_ends: bool = True,
    include_breadcrumbs: bool = True,
    include_segment_type_layers: bool = True,
    db_session=None,
) -> str:
    """
    Generate a spelunker link for a segment with all its endpoints.

    Args:
        project_name: Name of the project
        seed_id: Seed supervoxel ID (primary key)
        include_certain_ends: Whether to include certain endpoints layer. Defaults to True.
        include_uncertain_ends: Whether to include uncertain endpoints layer. Defaults to True.
        include_breadcrumbs: Whether to include breadcrumbs layer. Defaults to True.
        include_segment_type_layers: Whether to include segment type layers
            (mesh, mask, examples). Defaults to True.
        db_session: Optional database session to use. If not provided, creates a new one.

    Returns:
        Spelunker URL string
    """
    ng_state = get_segment_ng_state(
        project_name,
        seed_id,
        include_certain_ends=include_certain_ends,
        include_uncertain_ends=include_uncertain_ends,
        include_breadcrumbs=include_breadcrumbs,
        include_segment_type_layers=include_segment_type_layers,
        db_session=db_session,
    )
    encoded_state = urllib.parse.quote(json.dumps(ng_state), safe="")
    return f"https://spelunker.cave-explorer.org/#!{encoded_state}"
