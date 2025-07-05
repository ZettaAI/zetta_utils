"""
Helper functions for looking up segment IDs at given coordinates.

This module provides a unified function to get either:
1. The current segment ID at a coordinate (agglomerated)
2. The initial supervoxel ID at a coordinate (non-agglomerated)
"""

import os
from datetime import datetime, timezone
from typing import Any

import pcg_skel
from caveclient import CAVEclient

from zetta_utils import log
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.task_management.db.models import ProjectModel, SegmentModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.types import Segment

logger = log.get_logger()


def get_segment_id(
    project_name: str, coordinate: list[float], initial: bool = False, db_session: Any = None
) -> int:
    """
    Get the segment ID at the given coordinate for a project.

    Args:
        project_name: Name of the project
        coordinate: [x, y, z] coordinate at SV resolution
        initial: If True, get initial supervoxel ID. If False, get current agglomerated segment ID.
        db_session: Optional database session to use

    Returns:
        Segment ID at the coordinate
    """
    with get_session_context(db_session) as session:
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project '{project_name}' not found!")

        segmentation_path = project.segmentation_path
        sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

    voxel_coords = [int(coord) for coord in coordinate[:3]]

    logger.debug(
        f"Looking up {'initial' if initial else 'current'} segment at SV resolution "
        f"coordinates: {voxel_coords}"
    )

    layer = build_cv_layer(
        path=segmentation_path,
        cv_kwargs={"agglomerate": not initial},
        index_resolution=sv_resolution,
        default_desired_resolution=sv_resolution,
        allow_slice_rounding=True,
    )

    data = layer[
        voxel_coords[0] : voxel_coords[0] + 1,
        voxel_coords[1] : voxel_coords[1] + 1,
        voxel_coords[2] : voxel_coords[2] + 1,
    ]

    segment_id = int(data[0, 0, 0])
    logger.debug(f"{'Initial' if initial else 'Current'} segment ID: {segment_id}")

    return segment_id


def convert_to_sv_resolution(
    coordinate: list[float], from_resolution: list[int], sv_resolution: list[int]
) -> list[float]:
    scale_x = from_resolution[0] / sv_resolution[0]
    scale_y = from_resolution[1] / sv_resolution[1]
    scale_z = from_resolution[2] / sv_resolution[2]

    sv_coordinate = [coordinate[0] * scale_x, coordinate[1] * scale_y, coordinate[2] * scale_z]

    return sv_coordinate


def get_skeleton_length_mm(
    project_name: str,
    segment_id: int,
    server_address: str = "https://proofreading.zetta.ai",
    db_session: Any = None,
) -> float | None:
    """
    Get the skeleton length for a segment in millimeters.

    Args:
        project_name: Name of the project
        segment_id: The segment ID to get the skeleton for
        server_address: CAVE server address (default: "https://proofreading.zetta.ai")
        db_session: Optional database session to use

    Returns:
        Skeleton length in millimeters, or None if skeleton cannot be retrieved
    """
    # Get datastack name from project
    with get_session_context(db_session) as session:
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project '{project_name}' not found!")

        datastack_name = project.datastack_name
        if not datastack_name:
            raise ValueError(
                f"Project '{project_name}' does not have a datastack_name configured!"
            )

    try:
        # Initialize CAVE client
        client = CAVEclient(
            datastack_name=datastack_name,
            server_address=server_address,
            auth_token=os.getenv("CAVE_AUTH_TOKEN"),
        )

        # Get skeleton using pcg_skel
        logger.debug(f"Fetching skeleton for segment {segment_id}")
        skeleton = pcg_skel.pcg_skeleton(root_id=segment_id, client=client)

        # Calculate path length in nanometers
        skeleton_length_nm = skeleton.path_length()

        # Convert to millimeters
        skeleton_length_mm = float(skeleton_length_nm / 1_000_000)  # Convert to Python float

        logger.debug(f"Skeleton length for segment {segment_id}: {skeleton_length_mm:.2f} mm")
        return skeleton_length_mm

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to get skeleton for segment {segment_id}: {e}")
        return None


def update_segment_statistics(
    project_name: str,
    seed_id: int,
    server_address: str = "https://proofreading.zetta.ai",
    db_session: Any = None,
) -> dict:
    """
    Compute and update skeleton length and synapse counts for a segment.

    Args:
        project_name: Name of the project
        seed_id: Seed supervoxel ID (primary key)
        server_address: CAVE server address
        db_session: Optional database session to use

    Returns:
        Dictionary with updated statistics
    """
    with get_session_context(db_session) as session:
        # Get segment and project
        segment = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .first()
        )

        if not segment:
            raise ValueError(f"Segment with seed_id {seed_id} not found in project {project_name}")

        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project '{project_name}' not found!")

        if not project.datastack_name:
            raise ValueError(
                f"Project '{project_name}' does not have a datastack_name configured!"
            )

        if not project.synapse_table:
            raise ValueError(f"Project '{project_name}' does not have a synapse_table configured!")

        segment_id = segment.current_segment_id
        if not segment_id:
            logger.warning(f"Segment {seed_id} has no current_segment_id")
            return {"error": "No current_segment_id"}

        # Initialize CAVE client
        client = CAVEclient(
            datastack_name=project.datastack_name,
            server_address=server_address,
            auth_token=os.getenv("CAVE_AUTH_TOKEN"),
        )

        results: dict[str, Any] = {}

        # Get skeleton length
        try:
            logger.info(f"Computing skeleton length for segment {segment_id}")
            skeleton = pcg_skel.pcg_skeleton(root_id=segment_id, client=client)
            skeleton_length_nm = skeleton.path_length()
            skeleton_length_mm = float(skeleton_length_nm / 1_000_000)  # Convert to Python float
            segment.skeleton_path_length_mm = skeleton_length_mm
            results["skeleton_path_length_mm"] = skeleton_length_mm
            logger.info(f"Skeleton length: {skeleton_length_mm:.2f} mm")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to get skeleton: {e}")
            results["skeleton_error"] = str(e)

        # Get synapse counts
        try:
            logger.info(f"Computing synapse counts for segment {segment_id}")

            # Get pre-synaptic count
            pre_synapses = client.materialize.synapse_query(
                pre_ids=segment_id, synapse_table=project.synapse_table
            )
            pre_count = int(len(pre_synapses))  # Ensure Python int
            segment.pre_synapse_count = pre_count
            results["pre_synapse_count"] = pre_count

            # Get post-synaptic count
            post_synapses = client.materialize.synapse_query(
                post_ids=segment_id, synapse_table=project.synapse_table
            )
            post_count = int(len(post_synapses))  # Ensure Python int
            segment.post_synapse_count = post_count
            results["post_synapse_count"] = post_count

            logger.info(f"Synapse counts - Pre: {pre_count}, Post: {post_count}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to get synapse counts: {e}")
            results["synapse_error"] = str(e)

        # Update the segment in database
        session.commit()

        return results


def create_segment_from_coordinate(
    project_name: str,
    coordinate: list[float],
    batch: str | None = None,
    segment_type: str | None = None,
    expected_segment_type: str | None = None,
    extra_data: dict | None = None,
    db_session: Any = None,
) -> Segment:
    """
    Create a new segment given a coordinate.

    Args:
        project_name: Name of the project
        coordinate: [x, y, z] coordinate at SV resolution
        batch: Optional batch identifier
        segment_type: Optional segment type
        expected_segment_type: Optional expected segment type
        extra_data: Optional extra data dictionary
        db_session: Optional database session to use

    Returns:
        The created Segment

    Raises:
        ValueError: If project not found or segment already exists
    """
    # Get the initial supervoxel ID at the coordinate (seed_id)
    seed_id = get_segment_id(project_name, coordinate, initial=True, db_session=db_session)

    if seed_id == 0:
        raise ValueError(f"No supervoxel found at coordinate {coordinate}")

    # Get the current agglomerated segment ID
    current_segment_id = get_segment_id(
        project_name, coordinate, initial=False, db_session=db_session
    )

    with get_session_context(db_session) as session:
        # Check if segment already exists
        existing_segment = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .first()
        )

        if existing_segment:
            raise ValueError(
                f"Segment with seed_id {seed_id} already exists in project {project_name}"
            )

        # Create new segment
        now = datetime.now(timezone.utc)
        segment = SegmentModel(
            project_name=project_name,
            seed_id=seed_id,
            seed_x=float(coordinate[0]),
            seed_y=float(coordinate[1]),
            seed_z=float(coordinate[2]),
            current_segment_id=current_segment_id if current_segment_id != 0 else None,
            task_ids=[],
            batch=batch,
            segment_type=segment_type,
            expected_segment_type=expected_segment_type,
            status="WIP",
            is_exported=False,
            created_at=now,
            updated_at=now,
            extra_data=extra_data,
        )

        session.add(segment)
        session.commit()

        logger.info(f"Created segment with seed_id {seed_id} at coordinate {coordinate}")

        # Convert to Segment TypedDict
        segment_dict: Segment = {
            "project_name": project_name,
            "seed_id": seed_id,
            "seed_x": float(coordinate[0]),
            "seed_y": float(coordinate[1]),
            "seed_z": float(coordinate[2]),
            "current_segment_id": current_segment_id if current_segment_id != 0 else None,
            "task_ids": [],
            "status": "WIP",
            "is_exported": False,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # Add optional fields
        if batch is not None:
            segment_dict["batch"] = batch
        if segment_type is not None:
            segment_dict["segment_type"] = segment_type
        if expected_segment_type is not None:
            segment_dict["expected_segment_type"] = expected_segment_type
        if extra_data is not None:
            segment_dict["extra_data"] = extra_data

        return segment_dict
