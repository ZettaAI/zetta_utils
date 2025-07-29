"""
Get segment points for a neuron type using its seed mask bounds.

This module provides functionality to automatically query segments within a neuron type's
seed mask region and generate Neuroglancer visualizations.
"""

import json
import os
import urllib.parse
import uuid
from typing import Any

import click
import numpy as np
from scipy import ndimage
from scipy.ndimage import label

from zetta_utils import log
from zetta_utils.geometry.vec import Vec3D
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.task_management.db.models import (
    ProjectModel,
    SegmentModel,
    SegmentTypeModel,
)
from zetta_utils.task_management.db.session import get_session_context

logger = log.get_logger()


def _get_project_config(project_name: str, neuron_type_name: str) -> dict[str, Any]:
    """Get project and neuron type configuration from database."""
    with get_session_context() as session:
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project '{project_name}' not found in database")

        neuron_type = (
            session.query(SegmentTypeModel)
            .filter(
                SegmentTypeModel.project_name == project_name,
                SegmentTypeModel.type_name == neuron_type_name,
            )
            .first()
        )
        if not neuron_type:
            raise ValueError(
                f"Neuron type '{neuron_type_name}' not found in project '{project_name}'"
            )

        if not neuron_type.seed_mask:
            raise ValueError(f"Neuron type '{neuron_type_name}' has no seed_mask configured")

        return {
            "segmentation_path": project.segmentation_path,
            "sv_resolution": [
                project.sv_resolution_x,
                project.sv_resolution_y,
                project.sv_resolution_z,
            ],
            "extra_layers": project.extra_layers or [],
            "seed_mask_path": neuron_type.seed_mask,
            "region_mesh_path": neuron_type.region_mesh,
        }


def _get_mask_bounds_and_data(config: dict) -> tuple:
    """Get mask bounds and download segmentation/mask data."""
    logger.info("Loading mask layer to determine bounds...")
    mask_layer = build_cv_layer(
        path=config["seed_mask_path"],
        index_resolution=config["sv_resolution"],
        default_desired_resolution=config["sv_resolution"],
        allow_slice_rounding=True,
    )

    resolution: Vec3D = Vec3D(*config["sv_resolution"])
    mask_bounds = mask_layer.backend.get_bounds(resolution)
    bbox_min = (int(mask_bounds.start.x), int(mask_bounds.start.y), int(mask_bounds.start.z))
    bbox_max = (int(mask_bounds.stop.x), int(mask_bounds.stop.y), int(mask_bounds.stop.z))

    logger.info(f"Mask bounds: {bbox_min} to {bbox_max}")

    logger.info("Loading segmentation layer...")
    seg_layer = build_cv_layer(
        path=config["segmentation_path"],
        cv_kwargs={"agglomerate": True},
        index_resolution=config["sv_resolution"],
        default_desired_resolution=config["sv_resolution"],
        allow_slice_rounding=True,
    )

    logger.info("Downloading segmentation and mask data...")
    seg_data = seg_layer[
        bbox_min[0] : bbox_max[0], bbox_min[1] : bbox_max[1], bbox_min[2] : bbox_max[2]
    ]
    mask_data = mask_layer[
        bbox_min[0] : bbox_max[0], bbox_min[1] : bbox_max[1], bbox_min[2] : bbox_max[2]
    ]

    return bbox_min, bbox_max, seg_data, mask_data


def _process_segments_from_data(seg_data, mask_data, bbox_min, bbox_max, details: bool) -> dict:
    """Process segments and filter by mask overlap."""
    logger.info("Filtering segments by mask overlap...")
    unique_ids, counts = np.unique(seg_data, return_counts=True)
    mask = unique_ids != 0
    unique_ids = unique_ids[mask]
    counts = counts[mask]

    filtered_segments: dict[str, Any] = {}
    for seg_id, count in zip(unique_ids, counts):
        seg_mask = seg_data == seg_id
        overlap = np.any((seg_mask) & (mask_data > 0))

        if overlap:
            seg_mask_filtered = seg_mask & (mask_data > 0)
            if np.any(seg_mask_filtered):
                centroid = _calculate_segment_centroid(
                    seg_mask_filtered, seg_data.shape, bbox_min, bbox_max
                )
                if centroid:
                    if details:
                        filtered_segments[str(seg_id)] = {
                            "voxel_count": int(count),
                            "centroid": centroid,
                        }
                    else:
                        filtered_segments[str(seg_id)] = centroid

    logger.info(f"Found {len(filtered_segments)} segments after mask filtering")
    return filtered_segments


def _calculate_segment_centroid(seg_mask_filtered, shape, bbox_min, bbox_max) -> list[int] | None:
    """Calculate centroid for largest connected component."""
    labeled_components = label(seg_mask_filtered)
    if labeled_components.max() > 0:
        component_sizes = np.bincount(labeled_components.flat)[1:]
        largest_component_label = np.argmax(component_sizes) + 1
        largest_component_mask = labeled_components == largest_component_label
        centroid = ndimage.center_of_mass(largest_component_mask)

        array_z, array_x, array_y = shape[0], shape[1], shape[2]
        voxel_x = int(bbox_min[0] + (centroid[1] / array_x) * (bbox_max[0] - bbox_min[0]))
        voxel_y = int(bbox_min[1] + (centroid[2] / array_y) * (bbox_max[1] - bbox_min[1]))
        voxel_z = int(bbox_min[2] + (centroid[0] / array_z) * (bbox_max[2] - bbox_min[2]))
        return [voxel_x, voxel_y, voxel_z]
    return None


def _check_existing_segments_in_db(
    project_name: str, segments: dict, check_existing: bool, selection: str
) -> dict:
    """Check existing segments in database and return statistics."""
    existing_segments = 0
    new_segments = len(segments)
    existing_segment_ids = set()
    completed_segment_ids = set()

    if check_existing or selection != "all":
        logger.info("Checking existing segments in database...")
        with get_session_context() as session:
            segment_ids_for_query = [int(seg_id) for seg_id in segments]
            found_segments = (
                session.query(SegmentModel)
                .filter(
                    SegmentModel.project_name == project_name,
                    SegmentModel.current_segment_id.in_(segment_ids_for_query),
                )
                .all()
            )

            existing_segment_ids = {str(seg.current_segment_id) for seg in found_segments}
            completed_segment_ids = {
                str(seg.current_segment_id) for seg in found_segments if seg.status == "Completed"
            }
            existing_segments = len(existing_segment_ids)
            new_segments = len(segments) - existing_segments

            logger.info(f"Found {existing_segments} segments already in database")
            logger.info(f"Found {len(completed_segment_ids)} completed segments")
            logger.info(f"Found {new_segments} new segments")

    return {
        "existing_segments": existing_segments,
        "new_segments": new_segments,
        "existing_segment_ids": existing_segment_ids,
        "completed_segment_ids": completed_segment_ids,
    }


def _filter_by_selection(segments: dict, db_stats: dict, selection: str) -> dict:
    """Filter segments based on selection criteria."""
    if selection == "ingested":
        filtered = {
            seg_id: data
            for seg_id, data in segments.items()
            if seg_id in db_stats["existing_segment_ids"]
        }
        logger.info(f"Selection 'ingested': filtered to {len(filtered)} segments")
        return filtered
    elif selection == "not_ingested":
        filtered = {
            seg_id: data
            for seg_id, data in segments.items()
            if seg_id not in db_stats["existing_segment_ids"]
        }
        logger.info(f"Selection 'not_ingested': filtered to {len(filtered)} segments")
        return filtered
    elif selection == "completed":
        filtered = {
            seg_id: data
            for seg_id, data in segments.items()
            if seg_id in db_stats["completed_segment_ids"]
        }
        logger.info(f"Selection 'completed': filtered to {len(filtered)} segments")
        return filtered
    return segments


def get_seg_type_points(
    project_name: str,
    neuron_type_name: str,
    output_path: str | None = None,
    check_existing_segments: bool = True,
    debug: bool = False,
    details: bool = False,
    selection: str = "all",
    include_segments: bool = False,
) -> dict[str, Any]:
    """
    Get segment points for a neuron type and generate Neuroglancer visualization.

    Args:
        project_name: Name of the project
        neuron_type_name: Name of the neuron type
        output_path: Optional path to save JSON results
        check_existing_segments: Whether to check database for existing segments
        debug: If True, limit to 100 points for testing
        details: If True, include voxel_count and detailed data structure
        selection: Which segments to include - 'all', 'ingested', 'not_ingested', or 'completed'
        include_segments: If True, include selected segments in segmentation layer

    Returns:
        Dictionary with all results including NG link
    """
    logger.info(f"Getting segment points for {neuron_type_name} in {project_name}")

    # Get configuration
    config = _get_project_config(project_name, neuron_type_name)

    # Get bounds and data
    bbox_min, bbox_max, seg_data, mask_data = _get_mask_bounds_and_data(config)

    # Process segments
    filtered_segments = _process_segments_from_data(
        seg_data, mask_data, bbox_min, bbox_max, details
    )

    # Debug mode limiting
    if debug:
        filtered_segments = dict(list(filtered_segments.items())[:100])
        logger.info(f"Debug mode: limited to {len(filtered_segments)} segments")

    # Check database and filter by selection
    db_stats = _check_existing_segments_in_db(
        project_name, filtered_segments, check_existing_segments, selection
    )
    filtered_segments = _filter_by_selection(filtered_segments, db_stats, selection)

    # Build Neuroglancer state
    logger.info("Building Neuroglancer state...")
    ng_state = _build_neuroglancer_state(
        segmentation_path=config["segmentation_path"],
        seed_mask_path=config["seed_mask_path"],
        region_mesh_path=config["region_mesh_path"],
        extra_layers=config["extra_layers"],
        segment_points=filtered_segments,
        sv_resolution=config["sv_resolution"],
        include_segments=include_segments,
    )

    # Generate Neuroglancer link (using Spelunker)
    encoded_state = urllib.parse.quote(json.dumps(ng_state), safe="")
    ng_link = f"https://spelunker.cave-explorer.org/#!{encoded_state}"

    # Prepare results
    results = {
        "project_name": project_name,
        "neuron_type_name": neuron_type_name,
        "bounds": {"min": list(bbox_min), "max": list(bbox_max)},
        "segments": filtered_segments,
        "total_segments": len(filtered_segments),
        "existing_in_db": db_stats["existing_segments"],
        "new_segments": db_stats["new_segments"],
        "neuroglancer_link": ng_link,
    }

    # Save to JSON if path provided
    if output_path:
        abs_path = os.path.abspath(output_path)
        logger.info(f"Writing results to file: {abs_path}")
        print(f"Writing results to file: {abs_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {abs_path}")

    # Print summary
    print(f"\nResults for {neuron_type_name} in {project_name}:")
    print(f"  Total segments found: {len(filtered_segments)}")
    print(f"  Existing in database: {db_stats['existing_segments']}")
    print(f"  New segments: {db_stats['new_segments']}")
    print(f"  Bounds: {bbox_min} to {bbox_max}")
    print("\nNeuroglancer link:")
    print(ng_link)

    return results


def _build_neuroglancer_state(
    segmentation_path: str,
    seed_mask_path: str,
    region_mesh_path: str | None,
    extra_layers: list,
    segment_points: dict,
    sv_resolution: list[float],
    include_segments: bool = False,
) -> dict:
    """Build Neuroglancer state with all required layers and point annotations."""

    # Base state with proper NG configuration
    state: dict[str, Any] = {
        "dimensions": {
            "x": [sv_resolution[0] * 1e-9, "m"],
            "y": [sv_resolution[1] * 1e-9, "m"],
            "z": [sv_resolution[2] * 1e-9, "m"],
        },
        "crossSectionScale": 0.5,
        "projectionOrientation": [0.5, -0.5, 0.5, 0.5],
        "projectionScale": 2740.449487163767,
        "projectionDepth": 541651.3969244945,
        "layers": [],
        "showSlices": False,
        "selectedLayer": {"visible": True, "layer": "Segmentation"},
        "layout": "xy-3d",
    }

    # Add extra layers from project first
    if extra_layers:
        for layer_config in extra_layers:
            state["layers"].append(layer_config)

    # Add segmentation layer
    seg_layer: dict[str, Any] = {
        "type": "segmentation",
        "source": segmentation_path,
        "tab": "segments",
        "name": "Segmentation",
    }

    # Include selected segments if requested
    if include_segments and segment_points:
        seg_layer["segments"] = list(segment_points.keys())

    state["layers"].append(seg_layer)

    # Add seed mask layer as image
    state["layers"].append(
        {"type": "image", "source": seed_mask_path, "tab": "source", "name": "Seed Mask"}
    )

    # Add region mesh layer if available
    if region_mesh_path:
        state["layers"].append(
            {
                "type": "segmentation",
                "source": region_mesh_path,
                "tab": "rendering",
                "objectAlpha": 0.27,
                "segments": ["1"],
                "name": "Region Mesh",
            }
        )

    # Add point annotations for segments
    if segment_points:
        annotations: list[dict[str, Any]] = []
        for seg_data in segment_points.values():
            if isinstance(seg_data, dict):
                centroid = seg_data["centroid"]
            else:
                centroid = seg_data
            annotations.append(
                {"point": centroid, "type": "point", "id": str(uuid.uuid4()).replace("-", "")}
            )

        state["layers"].append(
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
                "annotationColor": "#00ffff",  # Cyan for segment points
                "annotations": annotations,
                "name": "Segment Points",
            }
        )

    return state


@click.command()
@click.argument("project_name")
@click.argument("neuron_type_name")
@click.option("--output", "-o", help="Path to save JSON results")
@click.option("--debug/--no-debug", default=False, help="Limit to 100 points for testing")
@click.option(
    "--check-existing/--no-check-existing",
    default=True,
    help="Check existing segments in database",
)
@click.option(
    "--details/--no-details", default=False, help="Include detailed data (voxel_count, etc.)"
)
@click.option(
    "--selection",
    type=click.Choice(["all", "ingested", "not_ingested", "completed"]),
    default="all",
    help="Which segments to include",
)
@click.option(
    "--include-segments/--no-include-segments",
    default=False,
    help="Include selected segments in segmentation layer",
)
def main(
    project_name: str,
    neuron_type_name: str,
    output: str | None,
    debug: bool,
    check_existing: bool,
    details: bool,
    selection: str,
    include_segments: bool,
):
    """Get segment points for a neuron type and generate Neuroglancer visualization."""
    try:
        get_seg_type_points(
            project_name=project_name,
            neuron_type_name=neuron_type_name,
            output_path=output,
            check_existing_segments=check_existing,
            debug=debug,
            details=details,
            selection=selection,
            include_segments=include_segments,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise click.ClickException(str(e))
