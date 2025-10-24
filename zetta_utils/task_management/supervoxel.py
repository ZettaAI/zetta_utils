"""Supervoxel management and update functions for tracking segment merges/splits."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db.models import SegmentEditEventModel, SupervoxelModel
from .db.session import get_session_context


def update_supervoxels_for_merge(  # pylint: disable=R0917
    old_root_ids: list[int],
    new_root_id: int,
    project_name: str,
    event_id: str,
    edit_timestamp: datetime,
    operation_type: str = "merge",
    db_session: Optional[Session] = None,
) -> int:
    """
    Update supervoxels table when segments are merged.

    Args:
        old_root_ids: List of old root IDs that were merged
        new_root_id: New root ID after merge
        project_name: Project name for event tracking
        event_id: Unique event ID for idempotency
        edit_timestamp: Timestamp of the edit operation
        operation_type: Type of operation (merge or split)
        db_session: Optional database session

    Returns:
        Number of supervoxels updated
    """
    with get_session_context(db_session) as session:
        existing_event = session.execute(
            select(SegmentEditEventModel).where(
                SegmentEditEventModel.project_name == project_name,
                SegmentEditEventModel.event_id == event_id,
            )
        ).scalar_one_or_none()

        if existing_event:
            return 0

        stmt = (
            update(SupervoxelModel)
            .where(SupervoxelModel.current_segment_id.in_(old_root_ids))
            .values(
                current_segment_id=new_root_id,
                updated_at=datetime.now(timezone.utc),
            )
        )
        result = session.execute(stmt)

        event = SegmentEditEventModel(
            project_name=project_name,
            event_id=event_id,
            old_root_ids=old_root_ids,
            new_root_ids=[new_root_id],  # Convert single ID to list for unified schema
            edit_timestamp=edit_timestamp,
            processed_at=datetime.now(timezone.utc),
            operation_type=operation_type,
        )
        session.add(event)
        session.commit()

        return result.rowcount


def update_supervoxels_for_split(  # pylint: disable=R0917
    old_root_id: int,
    new_root_ids: list[int],
    supervoxel_assignments: dict[int, int],
    project_name: str,
    event_id: str,
    edit_timestamp: datetime,
    operation_type: str = "split",
    db_session: Optional[Session] = None,
) -> int:
    """
    Update supervoxels table when a segment is split.

    Args:
        old_root_id: Original root ID that was split
        new_root_ids: List of new root IDs after split
        supervoxel_assignments: Dict mapping supervoxel_id to new_root_id
        project_name: Project name for event tracking
        event_id: Unique event ID for idempotency
        edit_timestamp: Timestamp of the edit operation
        operation_type: Type of operation (merge or split)
        db_session: Optional database session

    Returns:
        Number of supervoxels updated
    """
    with get_session_context(db_session) as session:
        existing_event = session.execute(
            select(SegmentEditEventModel).where(
                SegmentEditEventModel.project_name == project_name,
                SegmentEditEventModel.event_id == event_id,
            )
        ).scalar_one_or_none()

        if existing_event:
            return 0

        total_updated = 0
        for supervoxel_id, new_segment_id in supervoxel_assignments.items():
            stmt = (
                update(SupervoxelModel)
                .where(SupervoxelModel.supervoxel_id == supervoxel_id)
                .values(
                    current_segment_id=new_segment_id,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            result = session.execute(stmt)
            total_updated += result.rowcount

        event = SegmentEditEventModel(
            project_name=project_name,
            event_id=event_id,
            old_root_ids=[old_root_id],  # Convert single ID to list for unified schema
            new_root_ids=new_root_ids,
            edit_timestamp=edit_timestamp,
            processed_at=datetime.now(timezone.utc),
            operation_type=operation_type,
        )
        session.add(event)
        session.commit()

        return total_updated


def get_supervoxels_by_segment(
    segment_id: int,
    db_session: Optional[Session] = None,
) -> list[SupervoxelModel]:
    """
    Get all supervoxels belonging to a specific segment.

    Args:
        segment_id: Segment ID to query
        db_session: Optional database session

    Returns:
        List of supervoxel models
    """
    with get_session_context(db_session) as session:
        result = session.execute(
            select(SupervoxelModel).where(
                SupervoxelModel.current_segment_id == segment_id
            )
        )
        return list(result.scalars().all())


def create_supervoxel(  # pylint: disable=R0917
    supervoxel_id: int,
    seed_x: float,
    seed_y: float,
    seed_z: float,
    current_segment_id: int,
    db_session: Optional[Session] = None,
) -> SupervoxelModel:
    """
    Create a new supervoxel entry.

    Args:
        supervoxel_id: Unique supervoxel ID
        seed_x: X coordinate of seed point
        seed_y: Y coordinate of seed point
        seed_z: Z coordinate of seed point
        current_segment_id: Current segment ID
        db_session: Optional database session

    Returns:
        Created supervoxel model
    """
    with get_session_context(db_session) as session:
        now = datetime.now(timezone.utc)
        supervoxel = SupervoxelModel(
            supervoxel_id=supervoxel_id,
            seed_x=seed_x,
            seed_y=seed_y,
            seed_z=seed_z,
            current_segment_id=current_segment_id,
            created_at=now,
            updated_at=now,
        )
        session.add(supervoxel)
        session.commit()
        session.refresh(supervoxel)
        return supervoxel
