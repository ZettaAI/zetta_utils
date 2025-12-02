"""
Segment statistics update queue management functions.

This module handles queueing and processing segment statistics updates after PCG edits.
Updates skeleton_path_length_mm, pre_synapse_count, and post_synapse_count.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import and_, text
from sqlalchemy.dialects.postgresql import insert

from zetta_utils.task_management.db.models import SegmentModel, SegmentUpdateQueueModel
from zetta_utils.task_management.db.session import get_session_context

logger = logging.getLogger(__name__)


def queue_segment_updates_for_segments(
    project_name: str,
    segment_ids: list[int],
    db_session: Any = None,
) -> int:
    """
    Queue segment statistics updates for segments that have seed points.
    
    This function finds all segments (by seed_id) that are affected by the given
    segment_ids and queues them for skeleton length and synapse count updates.
    
    Args:
        project_name: Project name
        segment_ids: List of segment IDs that were affected by PCG edits
        db_session: Optional database session
        
    Returns:
        Number of segment statistics updates queued
    """
    with get_session_context(db_session) as session:
        print(f"Queueing skeleton updates for project {project_name}")
        print(f"Segment IDs: {segment_ids}")
        # Find all segments that have any of these segment_ids as current_segment_id
        affected_segments = (
            session.query(SegmentModel.seed_id, SegmentModel.current_segment_id)
            .filter(
                and_(
                    SegmentModel.project_name == project_name,
                    SegmentModel.current_segment_id.in_(segment_ids)
                )
            )
            .all()
        )

        if not affected_segments:
            print(f"No affected segments found for segment_ids {segment_ids}")
            logger.info(f"No segments found for segment_ids {segment_ids}")
            return 0

        print(f"Found {len(affected_segments)} segments to queue for skeleton updates")
        logger.info(f"Found {len(affected_segments)} segments to queue for skeleton updates")

        # Use PostgreSQL UPSERT to insert or update queue entries
        # This ensures we don't create duplicates and update existing entries
        queue_data = []
        now = datetime.now(timezone.utc)

        for segment in affected_segments:
            queue_data.append({
                "project_name": project_name,
                "seed_id": segment.seed_id,
                "current_segment_id": segment.current_segment_id,
                "status": "pending",
                "created_at": now,
                "last_attempt": None,
                "retry_count": 0,
                "error_message": None,
            })

        if queue_data:
            # Use PostgreSQL UPSERT (ON CONFLICT DO UPDATE)
            stmt = insert(SegmentUpdateQueueModel).values(queue_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["project_name", "seed_id"],
                set_={
                    "current_segment_id": stmt.excluded.current_segment_id,
                    "status": "pending",  # Reset to pending if already exists
                    "created_at": stmt.excluded.created_at,  # Update timestamp
                    "retry_count": 0,  # Reset retry count
                    "error_message": None,  # Clear any previous errors
                }
            )
            session.execute(stmt)
            session.commit()

            logger.info(f"Queued {len(queue_data)} skeleton updates for project {project_name}")
            return len(queue_data)

        # Defensive fallback: unreachable if affected_segments is non-empty
        # (queue_data mirrors affected_segments).
        return 0  # pragma: no cover


def get_next_pending_update(
    project_name: str,
    db_session: Any = None,
) -> Optional[dict]:
    """
    Get the next pending segment statistics update from the queue.
    
    Args:
        project_name: Project name
        db_session: Optional database session
        
    Returns:
        Dictionary with queue entry data, or None if no pending updates
    """
    with get_session_context(db_session) as session:
        queue_entry = (
            session.query(SegmentUpdateQueueModel)
            .filter(
                and_(
                    SegmentUpdateQueueModel.project_name == project_name,
                    SegmentUpdateQueueModel.status == "pending"
                )
            )
            .order_by(SegmentUpdateQueueModel.created_at.asc())
            .first()
        )

        if queue_entry:
            return queue_entry.to_dict()
        return None


def mark_update_processing(
    project_name: str,
    seed_id: int,
    db_session: Any = None,
) -> bool:
    """
    Mark a skeleton update as being processed.
    
    Args:
        project_name: Project name
        seed_id: Seed ID being processed
        db_session: Optional database session
        
    Returns:
        True if successfully marked as processing, False if entry not found
    """
    with get_session_context(db_session) as session:
        queue_entry = (
            session.query(SegmentUpdateQueueModel)
            .filter(
                and_(
                    SegmentUpdateQueueModel.project_name == project_name,
                    SegmentUpdateQueueModel.seed_id == seed_id
                )
            )
            .first()
        )

        if queue_entry:
            queue_entry.status = "processing"
            queue_entry.last_attempt = datetime.now(timezone.utc)
            session.commit()
            return True
        return False


def mark_update_completed(
    project_name: str,
    seed_id: int,
    db_session: Any = None,
) -> bool:
    """
    Mark a skeleton update as completed.
    
    Args:
        project_name: Project name
        seed_id: Seed ID that was successfully processed
        db_session: Optional database session
        
    Returns:
        True if successfully marked as completed, False if entry not found
    """
    with get_session_context(db_session) as session:
        queue_entry = (
            session.query(SegmentUpdateQueueModel)
            .filter(
                and_(
                    SegmentUpdateQueueModel.project_name == project_name,
                    SegmentUpdateQueueModel.seed_id == seed_id
                )
            )
            .first()
        )

        if queue_entry:
            queue_entry.status = "completed"
            queue_entry.last_attempt = datetime.now(timezone.utc)
            session.commit()
            return True
        return False


def mark_update_failed(
    project_name: str,
    seed_id: int,
    error_message: str,
    max_retries: int = 5,
    db_session: Any = None,
) -> bool:
    """
    Mark a skeleton update as failed and increment retry count.
    
    If retry count exceeds max_retries, marks as permanently failed.
    Otherwise, resets to pending for retry.
    
    Args:
        project_name: Project name
        seed_id: Seed ID that failed to process
        error_message: Error message to store
        max_retries: Maximum number of retries before permanent failure
        db_session: Optional database session
        
    Returns:
        True if successfully updated, False if entry not found
    """
    with get_session_context(db_session) as session:
        queue_entry = (
            session.query(SegmentUpdateQueueModel)
            .filter(
                and_(
                    SegmentUpdateQueueModel.project_name == project_name,
                    SegmentUpdateQueueModel.seed_id == seed_id
                )
            )
            .first()
        )

        if queue_entry:
            queue_entry.last_attempt = datetime.now(timezone.utc)
            queue_entry.retry_count += 1
            queue_entry.error_message = error_message

            if queue_entry.retry_count >= max_retries:
                queue_entry.status = "failed"
                logger.warning(
                    f"Skeleton update for seed {seed_id} permanently failed "
                    f"after {queue_entry.retry_count} attempts: {error_message}"
                )
            else:
                queue_entry.status = "pending"  # Retry
                logger.info(
                    f"Skeleton update for seed {seed_id} failed "
                    f"(attempt {queue_entry.retry_count}), will retry: {error_message}"
                )

            session.commit()
            return True
        return False


def get_queue_stats(project_name: str, db_session: Any = None) -> dict:
    """
    Get statistics about the skeleton update queue.
    
    Args:
        project_name: Project name
        db_session: Optional database session
        
    Returns:
        Dictionary with queue statistics
    """
    with get_session_context(db_session) as session:
        # Use raw SQL for efficient aggregation
        result = session.execute(
            text("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM segment_update_queue 
                WHERE project_name = :project_name
                GROUP BY status
            """),
            {"project_name": project_name}
        )

        stats = {"pending": 0, "processing": 0, "completed": 0, "failed": 0}
        for row in result:
            stats[row.status] = row.count

        return stats


def cleanup_completed_updates(
    project_name: str,
    days_old: int = 7,
    db_session: Any = None,
) -> int:
    """
    Remove completed skeleton update entries older than specified days.
    
    Args:
        project_name: Project name
        days_old: Remove completed entries older than this many days
        db_session: Optional database session
        
    Returns:
        Number of entries removed
    """
    with get_session_context(db_session) as session:
        cutoff_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)

        deleted_count = (
            session.query(SegmentUpdateQueueModel)
            .filter(
                and_(
                    SegmentUpdateQueueModel.project_name == project_name,
                    SegmentUpdateQueueModel.status == "completed",
                    SegmentUpdateQueueModel.last_attempt < cutoff_date
                )
            )
            .delete()
        )

        session.commit()
        logger.info(f"Cleaned up {deleted_count} completed skeleton update entries")
        return deleted_count
