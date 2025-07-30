"""
Generate Neuroglancer visualizations for segment types.

This module provides functionality to create Neuroglancer links showing segment types
with their associated meshes, masks, sample segments, and optionally completed segments.
"""

import json
import urllib.parse
from typing import Any

import click

from zetta_utils import log
from zetta_utils.task_management.db.models import ProjectModel, SegmentTypeModel
from zetta_utils.task_management.db.session import get_session_context

logger = log.get_logger()


def get_segment_type_layers(
    project_name: str, segment_type_name: str, include_names: bool = False, db_session: Any = None
) -> list[dict]:
    """
    Get Neuroglancer layers for a segment type.

    Args:
        project_name: Name of the project
        segment_type_name: Name of the segment type
        include_names: Whether to include segment type name in layer names
        db_session: Optional database session

    Returns:
        List of Neuroglancer layer configurations
    """
    logger.info(
        f"Building layers for segment type '{segment_type_name}' in project '{project_name}'"
    )

    with get_session_context(db_session) as session:
        # Get project configuration
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project '{project_name}' not found in database")

        # Get segment type configuration
        segment_type = (
            session.query(SegmentTypeModel)
            .filter(
                SegmentTypeModel.project_name == project_name,
                SegmentTypeModel.type_name == segment_type_name,
            )
            .first()
        )
        if not segment_type:
            raise ValueError(
                f"Segment type '{segment_type_name}' not found in project '{project_name}'"
            )

        # Extract project configuration
        segmentation_path = project.segmentation_path

        # Extract segment type configuration
        region_mesh_path = segment_type.region_mesh
        seed_mask_path = segment_type.seed_mask
        sample_segment_ids = segment_type.sample_segment_ids or []

    layers = []

    # Add main segmentation layer with sample segments (always include, but disabled by default)
    seg_layer_name = f"{segment_type_name} Examples" if include_names else "Segment Type Examples"
    seg_layer = {
        "type": "segmentation",
        "source": segmentation_path,
        "tab": "segments",
        "name": seg_layer_name,
        "visible": False,
    }

    # Include sample segments if they exist
    if sample_segment_ids:
        seg_layer["segments"] = sample_segment_ids
        logger.info(f"Including {len(sample_segment_ids)} sample segments in segmentation layer")
    else:
        logger.info("No sample segments found - examples layer will be empty")

    layers.append(seg_layer)

    # Add seed mask layer if available
    if seed_mask_path:
        mask_layer_name = (
            f"{segment_type_name} Seed Mask" if include_names else "Segment Type Seed Mask"
        )
        layers.append(
            {
                "type": "image",
                "source": seed_mask_path,
                "tab": "source",
                "name": mask_layer_name,
                "visible": False,
            }
        )
        logger.info(f"Added seed mask layer: {seed_mask_path}")

    # Add region mesh layer if available
    if region_mesh_path:
        mesh_layer_name = (
            f"{segment_type_name} Region Mesh" if include_names else "Segment Type Mesh"
        )
        layers.append(
            {
                "type": "segmentation",
                "source": region_mesh_path,
                "tab": "rendering",
                "objectAlpha": 0.27,
                "segments": ["1"],
                "name": mesh_layer_name,
                "visible": False,
            }
        )
        logger.info(f"Added region mesh layer: {region_mesh_path}")

    logger.info(f"Built {len(layers)} layers for segment type visualization")
    return layers


def get_segment_type_link(
    project_name: str,
    segment_type_name: str,
    include_completed_segments: bool = False,  # pylint: disable=unused-argument
    db_session: Any = None,
) -> str:
    """
    Get Neuroglancer link for a segment type.

    Args:
        project_name: Name of the project
        segment_type_name: Name of the segment type
        include_completed_segments: Whether to include completed segments as a separate layer
        db_session: Optional database session

    Returns:
        Neuroglancer (Spelunker) link URL
    """
    # Get project configuration for base state
    with get_session_context(db_session) as session:
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project '{project_name}' not found in database")

        sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

    # Get layers with names for links
    layers = get_segment_type_layers(
        project_name=project_name,
        segment_type_name=segment_type_name,
        include_names=True,
        db_session=db_session,
    )

    # Get project extra layers and add them first
    with get_session_context(db_session) as session:
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        extra_layers = project.extra_layers or []

    # Build complete layers list with extra layers first
    all_layers = []
    if extra_layers:
        for layer_config in extra_layers:
            all_layers.append(layer_config)
    all_layers.extend(layers)

    # Build Neuroglancer state
    state = {
        "dimensions": {
            "x": [sv_resolution[0] * 1e-9, "m"],
            "y": [sv_resolution[1] * 1e-9, "m"],
            "z": [sv_resolution[2] * 1e-9, "m"],
        },
        "crossSectionScale": 0.5,
        "projectionOrientation": [0.5, -0.5, 0.5, 0.5],
        "projectionScale": 2740.449487163767,
        "projectionDepth": 541651.3969244945,
        "layers": all_layers,
        "showSlices": False,
        "selectedLayer": {"visible": True, "layer": "Segmentation"},
        "layout": "xy-3d",
    }

    # Generate Neuroglancer link (using Spelunker)
    encoded_state = urllib.parse.quote(json.dumps(state), safe="")
    ng_link = f"https://spelunker.cave-explorer.org/#!{encoded_state}"

    logger.info(f"Generated Neuroglancer link for segment type '{segment_type_name}'")
    return ng_link


@click.command()
@click.argument("project_name")
@click.argument("segment_type_name")
@click.option(
    "--include-completed/--no-include-completed",
    default=False,
    help="Include completed segments as a separate layer",
)
def main(project_name: str, segment_type_name: str, include_completed: bool):
    """
    Generate Neuroglancer link for a segment type.

    Creates a visualization showing the segment type's region mesh (if available),
    seed mask (if available), sample segments, and optionally completed segments.
    """
    try:
        print(
            f"Generating Neuroglancer link for '{segment_type_name}' in "
            f"project '{project_name}'..."
        )

        ng_link = get_segment_type_link(
            project_name=project_name,
            segment_type_name=segment_type_name,
            include_completed_segments=include_completed,
        )

        print(f"\nNeuroglancer link for {segment_type_name}:")
        print(ng_link)

    except Exception as e:
        logger.error(f"Failed to generate link: {e}")
        raise click.ClickException(str(e))
