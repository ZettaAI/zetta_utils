"""
Generate neuroglancer links for segments with endpoints.

This module provides functions to create neuroglancer states and links
for segments in the database, including all their endpoints with proper
color coding.
"""

import json
import urllib.parse
import uuid
from typing import Any

from zetta_utils.task_management.db.models import (
    EndpointModel,
    ProjectModel,
    SegmentModel,
)
from zetta_utils.task_management.db.session import get_session_context


def get_segment_ng_state(
    project_name: str, seed_id: int, include_endpoints: bool = True, db_session=None
) -> dict:
    """
    Generate a neuroglancer state for a segment with all its endpoints.

    Args:
        project_name: Name of the project
        seed_id: Seed supervoxel ID (primary key)
        include_endpoints: Whether to include endpoint layers (certain/uncertain/breadcrumbs).
            Defaults to True.
        db_session: Optional database session to use. If not provided, creates a new one.

    Returns:
        Neuroglancer state dict with annotation layers for endpoints
    """

    def _generate_state(session):
        # Get segment
        segment = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .first()
        )

        if not segment:
            raise ValueError(f"Segment {seed_id} not found in project {project_name}")

        # Get project info
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project {project_name} not found")

        # Get all endpoints for this segment if needed
        if include_endpoints:
            endpoints = (
                session.query(EndpointModel)
                .filter_by(project_name=project_name, seed_id=seed_id)
                .all()
            )

            # Group endpoints by status
            certain_ends = []
            uncertain_ends = []
            breadcrumbs = []

            for endpoint in endpoints:
                point = [endpoint.x, endpoint.y, endpoint.z]
                annotation = {
                    "point": point,
                    "type": "point",
                    "id": str(uuid.uuid4()).replace("-", ""),
                }

                if endpoint.status == "CERTAIN":
                    certain_ends.append(annotation)
                elif endpoint.status == "UNCERTAIN":
                    uncertain_ends.append(annotation)
                elif endpoint.status == "BREADCRUMB":
                    breadcrumbs.append(annotation)

        # Build neuroglancer state
        sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

        ng_state: dict[str, Any] = {
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

        # Add extra layers first if configured in project
        if project.extra_layers and "layers" in project.extra_layers:
            for layer in project.extra_layers["layers"]:
                ng_state["layers"].append(layer)

        # Add segmentation layer
        ng_state["layers"].append(
            {
                "type": "segmentation",
                "source": project.segmentation_path,
                "tab": "segments",
                "segments": (
                    [str(segment.current_segment_id)] if segment.current_segment_id else []
                ),
                "name": "Segmentation",
            }
        )

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

        # Add root location annotation (green) - always show even if empty
        root_annotations = []
        if (
            segment.root_x is not None
            and segment.root_y is not None
            and segment.root_z is not None
        ):
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

        # Add endpoint layers only if requested
        if include_endpoints:
            # Add certain ends layer (yellow)
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
                    "annotationColor": "#ffff00",  # Yellow
                    "annotations": certain_ends,
                    "name": "Certain Ends",
                }
            )

            # Add uncertain ends layer (red)
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
                    "annotationColor": "#ff0000",  # Red
                    "annotations": uncertain_ends,
                    "name": "Uncertain Ends",
                }
            )

            # Add breadcrumbs layer (blue)
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
                    "tab": "rendering",
                    "annotationColor": "#0400ff",  # Blue
                    "annotations": breadcrumbs,
                    "name": "Breadcrumbs",
                }
            )

        return ng_state

    # Use provided session or create a new one
    if db_session is not None:
        return _generate_state(db_session)
    else:
        with get_session_context() as session:
            return _generate_state(session)


def get_segment_link(
    project_name: str, seed_id: int, include_endpoints: bool = True, db_session=None
) -> str:
    """
    Generate a spelunker link for a segment with all its endpoints.

    Args:
        project_name: Name of the project
        seed_id: Seed supervoxel ID (primary key)
        include_endpoints: Whether to include endpoint layers (certain/uncertain/breadcrumbs).
            Defaults to True.
        db_session: Optional database session to use. If not provided, creates a new one.

    Returns:
        Spelunker URL string
    """
    ng_state = get_segment_ng_state(
        project_name, seed_id, include_endpoints=include_endpoints, db_session=db_session
    )
    encoded_state = urllib.parse.quote(json.dumps(ng_state), safe="")
    return f"https://spelunker.cave-explorer.org/#!{encoded_state}"
