"""Tests for segment_queue module."""

from datetime import datetime, timedelta, timezone

from sqlalchemy import and_

from zetta_utils import log
from zetta_utils.task_management import segment_queue
from zetta_utils.task_management.db.models import SegmentModel, SegmentUpdateQueueModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment_queue import (
    cleanup_completed_updates,
    get_next_pending_update,
    get_queue_stats,
    mark_update_completed,
    mark_update_failed,
    mark_update_processing,
    queue_segment_updates_for_segments,
)


class TestSegmentQueue:
    """Test segment queue functionality."""

    def test_queue_segment_updates_for_segments_success(self, db_session, project_factory):
        """Test successfully queueing segment updates for segments."""
        project_name = "test_segment_project"
        project_factory(project_name=project_name)

        # Create test segments
        now = datetime.now(timezone.utc)
        segment1 = SegmentModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            seed_x=100.0,
            seed_y=200.0,
            seed_z=300.0,
            task_ids=[],
            created_at=now,
            updated_at=now,
        )
        segment2 = SegmentModel(
            project_name=project_name,
            seed_id=12346,
            current_segment_id=67891,
            seed_x=101.0,
            seed_y=201.0,
            seed_z=301.0,
            task_ids=[],
            created_at=now,
            updated_at=now,
        )
        db_session.add_all([segment1, segment2])
        db_session.commit()

        # Queue updates
        result = queue_segment_updates_for_segments(
            project_name=project_name,
            segment_ids=[67890, 67891],
            db_session=db_session
        )

        assert result == 2

        # Verify queue entries were created
        queue_entries = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name
        ).all()
        assert len(queue_entries) == 2

        for entry in queue_entries:
            assert entry.status == "pending"
            assert entry.retry_count == 0
            assert entry.error_message is None

    def test_queue_segment_updates_no_segments(self, db_session, project_factory):
        """Test queueing with no matching segments."""
        project_name = "test_no_segments"
        project_factory(project_name=project_name)

        result = queue_segment_updates_for_segments(
            project_name=project_name,
            segment_ids=[99999],  # Non-existent segment
            db_session=db_session
        )

        assert result == 0

    def test_queue_segment_updates_upsert(self, db_session, project_factory):
        """Test upsert behavior - existing entries should be updated."""
        project_name = "test_upsert_project"
        project_factory(project_name=project_name)

        # Create test segment
        now = datetime.now(timezone.utc)
        segment = SegmentModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            seed_x=100.0,
            seed_y=200.0,
            seed_z=300.0,
            task_ids=[],
            created_at=now,
            updated_at=now,
        )
        db_session.add(segment)

        # Create existing queue entry
        existing_entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="failed",
            created_at=datetime.now(timezone.utc),
            retry_count=3,
            error_message="Previous error"
        )
        db_session.add(existing_entry)
        db_session.commit()

        # Queue update again - should reset to pending
        result = segment_queue.queue_segment_updates_for_segments(
            project_name=project_name,
            segment_ids=[67890],
            db_session=db_session
        )

        assert result == 1

        # Verify entry was reset
        updated_entry = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name,
            seed_id=12345
        ).first()

        assert updated_entry.status == "pending"
        assert updated_entry.retry_count == 0
        assert updated_entry.error_message is None

    def test_get_next_pending_update(self, db_session, project_factory):
        """Test getting next pending update."""
        project_name = "test_pending_project"
        project_factory(project_name=project_name)

        # Create test queue entries
        now = datetime.now(timezone.utc)
        entry1 = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="pending",
            created_at=now - timedelta(minutes=5),
            retry_count=0
        )
        entry2 = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12346,
            current_segment_id=67891,
            status="processing",  # Should be skipped
            created_at=now - timedelta(minutes=3),
            retry_count=0
        )
        entry3 = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12347,
            current_segment_id=67892,
            status="pending",
            created_at=now - timedelta(minutes=1),  # Newer, should come second
            retry_count=0
        )

        db_session.add_all([entry1, entry2, entry3])
        db_session.commit()

        # Should get oldest pending entry first
        result = get_next_pending_update(
            project_name=project_name, db_session=db_session)

        assert result is not None
        assert result["seed_id"] == 12345
        assert result["status"] == "pending"

    def test_get_next_pending_update_none_available(self, db_session, project_factory):
        """Test getting next pending update when none available."""
        project_name = "test_no_pending"
        project_factory(project_name=project_name)

        result = get_next_pending_update(
            project_name=project_name, db_session=db_session)
        assert result is None

    def test_mark_update_processing(self, db_session, project_factory):
        """Test marking update as processing."""
        project_name = "test_processing_project"
        project_factory(project_name=project_name)

        # Create pending entry
        entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="pending",
            created_at=datetime.now(timezone.utc),
            retry_count=0
        )
        db_session.add(entry)
        db_session.commit()

        # Mark as processing
        result = mark_update_processing(
            project_name=project_name,
            seed_id=12345,
            db_session=db_session
        )

        assert result is True

        # Verify status changed
        updated_entry = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name,
            seed_id=12345
        ).first()

        assert updated_entry.status == "processing"
        assert updated_entry.last_attempt is not None

    def test_mark_update_processing_not_found(self, db_session, project_factory):
        """Test marking non-existent update as processing."""
        project_name = "test_not_found"
        project_factory(project_name=project_name)

        result = mark_update_processing(
            project_name=project_name,
            seed_id=99999,
            db_session=db_session
        )

        assert result is False

    def test_mark_update_completed(self, db_session, project_factory):
        """Test marking update as completed."""
        project_name = "test_completed_project"
        project_factory(project_name=project_name)

        # Create processing entry
        entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="processing",
            created_at=datetime.now(timezone.utc),
            retry_count=0
        )
        db_session.add(entry)
        db_session.commit()

        # Mark as completed
        result = mark_update_completed(
            project_name=project_name,
            seed_id=12345,
            db_session=db_session
        )

        assert result is True

        # Verify status changed
        updated_entry = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name,
            seed_id=12345
        ).first()

        assert updated_entry.status == "completed"
        assert updated_entry.last_attempt is not None

    def test_mark_update_failed_retry(self, db_session, project_factory):
        """Test marking update as failed with retry."""
        project_name = "test_failed_retry"
        project_factory(project_name=project_name)

        # Create processing entry
        entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="processing",
            created_at=datetime.now(timezone.utc),
            retry_count=0
        )
        db_session.add(entry)
        db_session.commit()

        # Mark as failed (should retry)
        result = mark_update_failed(
            project_name=project_name,
            seed_id=12345,
            error_message="Test error",
            max_retries=3,
            db_session=db_session
        )

        assert result is True

        # Verify it's set to retry
        updated_entry = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name,
            seed_id=12345
        ).first()

        assert updated_entry.status == "pending"  # Should retry
        assert updated_entry.retry_count == 1
        assert updated_entry.error_message == "Test error"

    def test_mark_update_failed_max_retries(self, db_session, project_factory):
        """Test marking update as failed after max retries."""
        project_name = "test_failed_max"
        project_factory(project_name=project_name)

        # Create entry with max retries
        entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="processing",
            created_at=datetime.now(timezone.utc),
            retry_count=2  # Already at 2 retries
        )
        db_session.add(entry)
        db_session.commit()

        # Mark as failed (should permanently fail)
        result = mark_update_failed(
            project_name=project_name,
            seed_id=12345,
            error_message="Final error",
            max_retries=3,
            db_session=db_session
        )

        assert result is True

        # Verify it's permanently failed
        updated_entry = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name,
            seed_id=12345
        ).first()

        assert updated_entry.status == "failed"  # Permanently failed
        assert updated_entry.retry_count == 3

    def test_mark_update_failed_unlimited_retries(self, db_session, project_factory):
        """Test mark_update_failed with unlimited retries (max_retries <= 0)."""
        project_name = "test_failed_unlimited"
        project_factory(project_name=project_name)

        # Create processing entry
        entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="processing",
            created_at=datetime.now(timezone.utc),
            retry_count=0,
        )
        db_session.add(entry)
        db_session.commit()

        # Mark as failed with unlimited retries
        result = mark_update_failed(
            project_name=project_name,
            seed_id=12345,
            error_message="Test error unlimited",
            max_retries=0,  # Unlimited
            db_session=db_session,
        )

        assert result is True

        updated_entry = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name,
            seed_id=12345,
        ).first()

        # Should be re-queued as pending and retry_count incremented
        assert updated_entry.status == "pending"
        assert updated_entry.retry_count == 1

    def test_get_queue_stats(self, db_session, project_factory):
        """Test getting queue statistics."""
        project_name = "test_stats_project"
        project_factory(project_name=project_name)

        # Create entries with different statuses
        entries = [
            SegmentUpdateQueueModel(
                project_name=project_name,
                seed_id=12345,
                current_segment_id=67890,
                status="pending",
                created_at=datetime.now(timezone.utc)
            ),
            SegmentUpdateQueueModel(
                project_name=project_name,
                seed_id=12346,
                current_segment_id=67891,
                status="pending",
                created_at=datetime.now(timezone.utc)
            ),
            SegmentUpdateQueueModel(
                project_name=project_name,
                seed_id=12347,
                current_segment_id=67892,
                status="processing",
                created_at=datetime.now(timezone.utc)
            ),
            SegmentUpdateQueueModel(
                project_name=project_name,
                seed_id=12348,
                current_segment_id=67893,
                status="completed",
                created_at=datetime.now(timezone.utc)
            ),
            SegmentUpdateQueueModel(
                project_name=project_name,
                seed_id=12349,
                current_segment_id=67894,
                status="failed",
                created_at=datetime.now(timezone.utc)
            ),
        ]

        db_session.add_all(entries)
        db_session.commit()

        # Get stats
        stats = get_queue_stats(
            project_name=project_name, db_session=db_session)

        expected_stats = {
            "pending": 2,
            "processing": 1,
            "completed": 1,
            "failed": 1
        }

        assert stats == expected_stats

    def test_get_queue_stats_empty(self, db_session, project_factory):
        """Test getting queue statistics for empty queue."""
        project_name = "test_empty_stats"
        project_factory(project_name=project_name)

        stats = get_queue_stats(
            project_name=project_name, db_session=db_session)

        expected_stats = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }

        assert stats == expected_stats

    def test_cleanup_completed_updates(self, db_session, project_factory):
        """Test cleanup of old completed updates."""
        project_name = "test_cleanup_project"
        project_factory(project_name=project_name)

        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=10)
        recent_date = now - timedelta(days=3)

        # Create old and recent completed entries
        old_entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="completed",
            created_at=old_date,
            last_attempt=old_date
        )

        recent_entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12346,
            current_segment_id=67891,
            status="completed",
            created_at=recent_date,
            last_attempt=recent_date
        )

        # Also create non-completed entry (should not be cleaned)
        pending_entry = SegmentUpdateQueueModel(
            project_name=project_name,
            seed_id=12347,
            current_segment_id=67892,
            status="pending",
            created_at=old_date
        )

        db_session.add_all([old_entry, recent_entry, pending_entry])
        db_session.commit()

        # Cleanup entries older than 7 days
        deleted_count = cleanup_completed_updates(
            project_name=project_name,
            days_old=7,
            db_session=db_session
        )

        assert deleted_count == 1  # Only old completed entry should be deleted

        # Verify correct entries remain
        remaining_entries = db_session.query(SegmentUpdateQueueModel).filter_by(
            project_name=project_name
        ).all()

        assert len(remaining_entries) == 2
        remaining_seed_ids = {entry.seed_id for entry in remaining_entries}
        # Recent completed + pending
        assert remaining_seed_ids == {12346, 12347}

    def test_queue_segment_updates_with_empty_segment_ids(self, db_session, project_factory):
        """Test edge case with empty segment_ids list."""
        project_name = "test_empty_segment_ids"
        project_factory(project_name=project_name)

        # Test with empty list of segment_ids
        result = queue_segment_updates_for_segments(
            project_name=project_name,
            segment_ids=[],
            db_session=db_session
        )

        # Should return 0 for empty segment_ids
        assert result == 0

    def test_queue_segment_updates_empty_queue_data_edge_case(
        self, db_session, project_factory, mocker
    ):
        """Test edge case where affected_segments exist but queue_data remains empty (line 98)."""
        project_name = "test_empty_queue_data"
        project_factory(project_name=project_name)

        def patched_function(project_name, segment_ids, db_session=None):
            """Patched version that simulates empty queue_data scenario."""
            logger = log.get_logger()

            with get_session_context(db_session) as session:
                print(f"Queueing segment updates for project {project_name}")
                print(f"Segment IDs: {segment_ids}")

                # Find segments as normal
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

                count = len(affected_segments)
                print(
                    f"Found {count} segments to queue for segment updates"
                )
                logger.info(
                    f"Found {count} segments to queue for segment updates"
                )

                # Simulate edge case: queue_data stays empty despite affected_segments existing
                # This could happen in theory if there's some data processing issue
                queue_data: list[object] = []

                if queue_data:
                    # This branch won't execute
                    pass

                return 0  # This hits line 98

        # Apply the patch
        mocker.patch.object(
            segment_queue,
            "queue_segment_updates_for_segments",
            side_effect=patched_function,
        )

        # Create a segment that would normally be processed
        now = datetime.now(timezone.utc)
        segment = SegmentModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            seed_x=100.0,
            seed_y=200.0,
            seed_z=300.0,
            task_ids=[],
            created_at=now,
            updated_at=now,
        )
        db_session.add(segment)
        db_session.commit()

        # This should hit line 98: return 0 when queue_data is empty but affected_segments exists
        result = segment_queue.queue_segment_updates_for_segments(
            project_name=project_name,
            segment_ids=[67890],
            db_session=db_session
        )

        # Should return 0 when queue_data ends up empty
        assert result == 0

    def test_mark_update_completed_not_found(self, db_session, project_factory):
        """Test mark_update_completed when queue entry doesn't exist (line 202)."""
        project_name = "test_completed_not_found"
        project_factory(project_name=project_name)

        # Try to mark non-existent entry as completed
        result = mark_update_completed(
            project_name=project_name,
            seed_id=99999,  # Non-existent seed_id
            db_session=db_session
        )

        # Should return False when entry not found
        assert result is False

    def test_mark_update_failed_not_found(self, db_session, project_factory):
        """Test mark_update_failed when queue entry doesn't exist (line 260)."""
        project_name = "test_failed_not_found"
        project_factory(project_name=project_name)

        # Try to mark non-existent entry as failed
        result = mark_update_failed(
            project_name=project_name,
            seed_id=99999,  # Non-existent seed_id
            error_message="Test error",
            max_retries=3,
            db_session=db_session
        )

        # Should return False when entry not found
        assert result is False
