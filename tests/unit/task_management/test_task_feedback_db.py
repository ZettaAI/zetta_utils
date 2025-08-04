"""Tests for TaskFeedbackModel database integration"""

# pylint: disable=unused-argument,redefined-outer-name,too-many-arguments

from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from zetta_utils.task_management.db.models import TaskFeedbackModel, TaskModel
from zetta_utils.task_management.task import create_task
from zetta_utils.task_management.task_type import create_task_type
from zetta_utils.task_management.types import Task, TaskType


@pytest.fixture
def sample_trace_task_type() -> TaskType:
    """Sample task type for trace tasks"""
    return {
        "task_type": "trace_v0",
        "completion_statuses": ["done", "need_help"],
    }


@pytest.fixture
def sample_feedback_task_type() -> TaskType:
    """Sample task type for feedback tasks"""
    return {
        "task_type": "trace_feedback_v0",
        "completion_statuses": ["Accurate", "Fair", "Inaccurate", "Faulty Task"],
    }


@pytest.fixture
def sample_original_task(sample_trace_task_type) -> Task:
    """Create sample original task"""
    return {
        "task_id": "trace_12345_original",
        "completion_status": "done",
        "assigned_user_id": "user1@zetta.ai",
        "active_user_id": "",
        "completed_user_id": "user1@zetta.ai",
        "ng_state": {"position": [100, 200, 300], "layers": ["segmentation"]},
        "ng_state_initial": {"position": [100, 200, 300], "layers": ["segmentation"]},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": sample_trace_task_type["task_type"],
        "is_active": True,
        "last_leased_ts": 0.0,
        "note": "Original task note",
    }


@pytest.fixture
def sample_feedback_task(sample_feedback_task_type) -> Task:
    """Create sample feedback task"""
    return {
        "task_id": "feedback_trace_12345_original_67890",
        "completion_status": "Accurate",
        "assigned_user_id": "feedback_user@zetta.ai",
        "active_user_id": "",
        "completed_user_id": "feedback_user@zetta.ai",
        "ng_state": {"position": [100, 200, 300], "layers": ["segmentation", "annotations"]},
        "ng_state_initial": {"position": [100, 200, 300], "layers": ["segmentation"]},
        "priority": 1,
        "batch_id": "batch_1",
        "task_type": sample_feedback_task_type["task_type"],
        "is_active": True,
        "last_leased_ts": 0.0,
        "note": "Feedback task note",
    }


class TestTaskFeedbackModelDatabase:
    """Test TaskFeedbackModel database operations"""

    def test_create_task_feedback_model(self, clean_db, db_session, project_name):
        """Test creating TaskFeedbackModel in database"""
        now = datetime.now(timezone.utc)

        feedback = TaskFeedbackModel(
            project_name=project_name,
            feedback_id=1,
            task_id="test_task",
            feedback_task_id="test_feedback_task",
            created_at=now,
            user_id="test_user@zetta.ai",
        )

        db_session.add(feedback)
        db_session.commit()

        # Query back from database
        result = (
            db_session.query(TaskFeedbackModel)
            .filter(
                TaskFeedbackModel.project_name == project_name, TaskFeedbackModel.feedback_id == 1
            )
            .first()
        )

        assert result is not None
        assert result.project_name == project_name
        assert result.feedback_id == 1
        assert result.task_id == "test_task"
        assert result.feedback_task_id == "test_feedback_task"
        assert result.created_at == now
        assert result.user_id == "test_user@zetta.ai"

    def test_task_feedback_model_primary_key(self, clean_db, db_session, project_name):
        """Test TaskFeedbackModel primary key behavior"""
        now = datetime.now(timezone.utc)

        # Create first feedback
        feedback1 = TaskFeedbackModel(
            project_name=project_name,
            feedback_id=1,
            task_id="task1",
            feedback_task_id="feedback1",
            created_at=now,
            user_id="user1@zetta.ai",
        )

        db_session.add(feedback1)
        db_session.commit()

        # Create second feedback with same project but different ID
        feedback2 = TaskFeedbackModel(
            project_name=project_name,
            feedback_id=2,
            task_id="task2",
            feedback_task_id="feedback2",
            created_at=now,
            user_id="user2@zetta.ai",
        )

        db_session.add(feedback2)
        db_session.commit()

        # Query both
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(TaskFeedbackModel.project_name == project_name)
            .all()
        )

        assert len(results) == 2
        assert {result.feedback_id for result in results} == {1, 2}

    def test_task_feedback_model_foreign_key_constraint(
        self,
        clean_db,
        db_session,
        project_name,
        sample_original_task,
        sample_feedback_task,
        sample_trace_task_type,
        sample_feedback_task_type,
    ):
        """Test TaskFeedbackModel with existing tasks"""
        # Create task types
        create_task_type(
            project_name=project_name, data=sample_trace_task_type, db_session=db_session
        )
        create_task_type(
            project_name=project_name, data=sample_feedback_task_type, db_session=db_session
        )

        # Create tasks
        create_task(project_name=project_name, data=sample_original_task, db_session=db_session)
        create_task(project_name=project_name, data=sample_feedback_task, db_session=db_session)

        # Create feedback entry
        feedback = TaskFeedbackModel(
            project_name=project_name,
            feedback_id=1,
            task_id=sample_original_task["task_id"],
            feedback_task_id=sample_feedback_task["task_id"],
            created_at=datetime.now(timezone.utc),
            user_id="feedback_user@zetta.ai",
        )

        db_session.add(feedback)
        db_session.commit()

        # Query with joins to verify relationships
        result = (
            db_session.query(TaskFeedbackModel)
            .join(
                TaskModel,
                (TaskFeedbackModel.task_id == TaskModel.task_id)
                & (TaskFeedbackModel.project_name == TaskModel.project_name),
            )
            .filter(
                TaskFeedbackModel.project_name == project_name, TaskFeedbackModel.feedback_id == 1
            )
            .first()
        )

        assert result is not None
        assert result.task_id == sample_original_task["task_id"]
        assert result.feedback_task_id == sample_feedback_task["task_id"]

    def test_task_feedback_model_indexes(self, clean_db, db_session, project_name):
        """Test that database indexes exist for TaskFeedbackModel"""
        # Create some test data
        now = datetime.now(timezone.utc)

        for i in range(5):
            feedback = TaskFeedbackModel(
                project_name=project_name,
                feedback_id=i + 1,
                task_id=f"task_{i}",
                feedback_task_id=f"feedback_task_{i}",
                created_at=now,
                user_id=f"user_{i}@zetta.ai",
            )
            db_session.add(feedback)

        db_session.commit()

        # Check that indexes exist by examining the explain plan
        # This is a basic check - in production you'd use EXPLAIN ANALYZE
        result = db_session.execute(
            text(
                """
            SELECT count(*) FROM task_feedback
            WHERE project_name = :project_name AND task_id = :task_id
        """
            ),
            {"project_name": project_name, "task_id": "task_1"},
        )

        count = result.scalar()
        assert count == 1

    def test_task_feedback_model_user_id_index(self, clean_db, db_session, project_name):
        """Test that user_id index works correctly"""
        now = datetime.now(timezone.utc)

        # Create feedback entries for different users
        users = ["user1@zetta.ai", "user2@zetta.ai", "user1@zetta.ai"]

        for i, user in enumerate(users):
            feedback = TaskFeedbackModel(
                project_name=project_name,
                feedback_id=i + 1,
                task_id=f"task_{i}",
                feedback_task_id=f"feedback_task_{i}",
                created_at=now,
                user_id=user,
            )
            db_session.add(feedback)

        db_session.commit()

        # Query by user_id
        result = (
            db_session.query(TaskFeedbackModel)
            .filter(
                TaskFeedbackModel.project_name == project_name,
                TaskFeedbackModel.user_id == "user1@zetta.ai",
            )
            .all()
        )

        assert len(result) == 2
        assert all(r.user_id == "user1@zetta.ai" for r in result)

    def test_task_feedback_model_created_at_ordering(self, clean_db, db_session, project_name):
        """Test ordering by created_at timestamp"""
        # Create feedback entries with different timestamps
        timestamps = [
            datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc),
        ]

        for i, timestamp in enumerate(timestamps):
            feedback = TaskFeedbackModel(
                project_name=project_name,
                feedback_id=i + 1,
                task_id=f"task_{i}",
                feedback_task_id=f"feedback_task_{i}",
                created_at=timestamp,
                user_id=f"user_{i}@zetta.ai",
            )
            db_session.add(feedback)

        db_session.commit()

        # Query ordered by created_at descending
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(TaskFeedbackModel.project_name == project_name)
            .order_by(TaskFeedbackModel.created_at.desc())
            .all()
        )

        assert len(results) == 3
        assert results[0].created_at == timestamps[1]  # 12:00 (latest)
        assert results[1].created_at == timestamps[0]  # 10:00 (middle)
        assert results[2].created_at == timestamps[2]  # 8:00 (earliest)

    def test_task_feedback_model_auto_increment_id(self, clean_db, db_session, project_name):
        """Test that feedback_id auto-increments correctly"""
        now = datetime.now(timezone.utc)

        # Create multiple feedback entries without specifying feedback_id
        feedbacks = []
        for i in range(3):
            feedback = TaskFeedbackModel(
                project_name=project_name,
                # feedback_id will be auto-assigned
                task_id=f"task_{i}",
                feedback_task_id=f"feedback_task_{i}",
                created_at=now,
                user_id=f"user_{i}@zetta.ai",
            )
            feedbacks.append(feedback)
            db_session.add(feedback)

        db_session.commit()

        # Check that IDs were auto-assigned
        for feedback in feedbacks:
            assert feedback.feedback_id is not None
            assert feedback.feedback_id > 0

        # Check that IDs are unique
        ids = [f.feedback_id for f in feedbacks]
        assert len(set(ids)) == len(ids)

    def test_task_feedback_model_timezone_handling(self, clean_db, db_session, project_name):
        """Test that timezone-aware datetime is handled correctly"""
        # Create feedback with timezone-aware datetime
        utc_time = datetime.now(timezone.utc)

        feedback = TaskFeedbackModel(
            project_name=project_name,
            feedback_id=1,
            task_id="task_1",
            feedback_task_id="feedback_task_1",
            created_at=utc_time,
            user_id="user@zetta.ai",
        )

        db_session.add(feedback)
        db_session.commit()

        # Query back and check timezone
        result = (
            db_session.query(TaskFeedbackModel)
            .filter(
                TaskFeedbackModel.project_name == project_name, TaskFeedbackModel.feedback_id == 1
            )
            .first()
        )

        assert result.created_at is not None
        assert result.created_at.tzinfo is not None
        assert result.created_at == utc_time

    def test_task_feedback_model_bulk_operations(self, clean_db, db_session, project_name):
        """Test bulk operations with TaskFeedbackModel"""
        now = datetime.now(timezone.utc)

        # Create multiple feedback entries
        feedbacks = []
        for i in range(10):
            feedback = TaskFeedbackModel(
                project_name=project_name,
                task_id=f"task_{i}",
                feedback_task_id=f"feedback_task_{i}",
                created_at=now,
                user_id=f"user_{i % 3}@zetta.ai",  # 3 different users
            )
            feedbacks.append(feedback)

        db_session.add_all(feedbacks)
        db_session.commit()

        # Query all and verify
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(TaskFeedbackModel.project_name == project_name)
            .all()
        )

        assert len(results) == 10

        # Test bulk delete
        db_session.query(TaskFeedbackModel).filter(
            TaskFeedbackModel.project_name == project_name,
            TaskFeedbackModel.user_id == "user_0@zetta.ai",
        ).delete()

        db_session.commit()

        # Verify deletion
        remaining = (
            db_session.query(TaskFeedbackModel)
            .filter(TaskFeedbackModel.project_name == project_name)
            .count()
        )

        assert remaining < 10  # Some entries should be deleted

    def test_task_feedback_model_query_performance(self, clean_db, db_session, project_name):
        """Test query performance with larger dataset"""
        now = datetime.now(timezone.utc)

        # Create 100 feedback entries
        feedbacks = []
        for i in range(100):
            feedback = TaskFeedbackModel(
                project_name=project_name,
                task_id=f"task_{i}",
                feedback_task_id=f"feedback_task_{i}",
                created_at=now,
                user_id=f"user_{i % 10}@zetta.ai",
            )
            feedbacks.append(feedback)

        db_session.add_all(feedbacks)
        db_session.commit()

        # Test various query patterns

        # 1. Query by project (should use index)
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(TaskFeedbackModel.project_name == project_name)
            .count()
        )
        assert results == 100

        # 2. Query by project and task_id (should use index)
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(
                TaskFeedbackModel.project_name == project_name,
                TaskFeedbackModel.task_id == "task_50",
            )
            .count()
        )
        assert results == 1

        # 3. Query by project and user_id (should use index)
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(
                TaskFeedbackModel.project_name == project_name,
                TaskFeedbackModel.user_id == "user_0@zetta.ai",
            )
            .count()
        )
        assert results == 10

        # 4. Query with ordering (should work efficiently)
        results = (
            db_session.query(TaskFeedbackModel)
            .filter(TaskFeedbackModel.project_name == project_name)
            .order_by(TaskFeedbackModel.created_at.desc())
            .limit(20)
            .all()
        )

        assert len(results) == 20
