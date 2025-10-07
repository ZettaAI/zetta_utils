"""
Helper functions for looking up segment IDs at given coordinates.

This module provides a unified function to get either:
1. The current segment ID at a coordinate (agglomerated)
2. The initial supervoxel ID at a coordinate (non-agglomerated)
"""

import os
import time
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

# Global cache for CAVE clients
_cave_client_cache: dict[str, CAVEclient] = {}


def get_cave_client(
    datastack_name: str, server_address: str = "https://proofreading.zetta.ai"
) -> CAVEclient:
    """Get or create a cached CAVE client to avoid too many connections."""
    cache_key = f"{datastack_name}@{server_address}"

    if cache_key not in _cave_client_cache:
        logger.debug(f"Creating new CAVE client for {cache_key}")
        _cave_client_cache[cache_key] = CAVEclient(
            datastack_name=datastack_name,
            server_address=server_address,
            auth_token=os.getenv("CAVE_AUTH_TOKEN"),
        )

    return _cave_client_cache[cache_key]


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
        # Get cached CAVE client
        client = get_cave_client(datastack_name, server_address)

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


def update_segment_statistics(  # pylint: disable=too-many-statements
    project_name: str,
    seed_id: int,
    server_address: str = "https://proofreading.zetta.ai",
    db_session: Any = None,
) -> dict:
    """
    Update current segment ID and compute skeleton length and synapse counts for a segment.

    This function first updates the current_segment_id by querying the seed location to get
    the latest valid segment ID, then computes skeleton length and synapse counts using
    live queries to ensure the most current data.

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

        results: dict[str, Any] = {}

        # First, update current segment ID from seed location to get latest valid ID
        try:
            logger.info(f"Updating current segment ID for seed {seed_id}")
            coordinate = [segment.seed_x, segment.seed_y, segment.seed_z]
            current_id = get_segment_id(
                project_name=project_name, coordinate=coordinate, initial=False, db_session=session
            )

            if current_id and current_id > 0:
                old_id = segment.current_segment_id
                segment.current_segment_id = current_id
                segment.updated_at = datetime.now(timezone.utc)
                logger.info(f"Updated current segment ID: {old_id} -> {current_id}")
            else:
                logger.warning(f"No segment found at seed location for seed {seed_id}")
                return {"error": "No segment found at seed location"}

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to update current segment ID: {e}")
            return {"error": f"Failed to update current segment ID: {e}"}

        segment_id = segment.current_segment_id
        if not segment_id:
            logger.warning(f"Segment {seed_id} has no current_segment_id")
            return {"error": "No current_segment_id"}

        # Get cached CAVE client
        client = get_cave_client(project.datastack_name, server_address)

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

        # Get synapse counts using live query
        try:
            logger.info(f"Computing synapse counts for segment {segment_id}")
            current_time = datetime.now(timezone.utc)

            # Get pre-synaptic count (live query)
            pre_df = client.materialize.live_query(
                project.synapse_table,
                current_time,
                filter_equal_dict={"pre_pt_root_id": segment_id},
            )
            pre_count = int(len(pre_df))  # Ensure Python int
            segment.pre_synapse_count = pre_count
            results["pre_synapse_count"] = pre_count

            # Get post-synaptic count (live query)
            post_df = client.materialize.live_query(
                project.synapse_table,
                current_time,
                filter_equal_dict={"post_pt_root_id": segment_id},
            )
            post_count = int(len(post_df))  # Ensure Python int
            segment.post_synapse_count = post_count
            results["post_synapse_count"] = post_count

            logger.info(f"Synapse counts - Pre: {pre_count}, Post: {post_count}")

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to get synapse counts: {e}")
            results["synapse_error"] = str(e)

        # Update the segment in database
        session.commit()

        return results


def update_segment_info(
    project_name: str,
    seed_id: int,
    server_address: str = "https://proofreading.zetta.ai",
    db_session: Any = None,
) -> dict:
    """
    Update segment's current ID from seed location and statistics.

    Args:
        project_name: Name of the project
        seed_id: Seed supervoxel ID
        server_address: CAVE server address
        db_session: Optional database session to use

    Returns:
        Dictionary with updated statistics
    """

    with get_session_context(db_session) as session:
        # Get segment
        segment = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .first()
        )

        if not segment:
            raise ValueError(f"Segment with seed_id {seed_id} not found in project {project_name}")

        # Update current segment ID from seed location
        coordinate = [segment.seed_x, segment.seed_y, segment.seed_z]
        current_id = get_segment_id(
            project_name=project_name, coordinate=coordinate, initial=False, db_session=session
        )

        if current_id and current_id > 0:
            segment.current_segment_id = current_id
            segment.updated_at = datetime.now(timezone.utc)
            session.commit()
        else:
            return {"error": "No segment at seed location"}

    # Now update statistics with the new segment ID, with retry logic for rate limiting
    max_retries = 5
    retry_delay: float = 10  # Start with 10 seconds as requested

    for attempt in range(max_retries):
        try:
            return update_segment_statistics(project_name, seed_id, server_address, db_session)
        except (ValueError, RuntimeError, IOError) as e:
            error_str = str(e)
            # Check for rate limiting error (429)
            if "429 Too Many Requests" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limited on segment {seed_id}, sleeping {retry_delay}s "
                        f"before retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 60)  # Exponential backoff up to 60s
                    continue
                logger.error(f"Rate limited on segment {seed_id} after {max_retries} attempts")
                return {"error": f"Rate limited after {max_retries} attempts"}
            # Re-raise non-rate-limit errors
            raise

    # This should never be reached due to the raise above, but needed for type checking
    return {"error": "Unexpected error"}


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
            status="Raw",
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
            "status": "Raw",
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
