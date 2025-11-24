"""Tests for SkeletonUpdateQueueModel."""

from datetime import datetime, timezone

from zetta_utils.task_management.db.models import SkeletonUpdateQueueModel


class TestSkeletonUpdateQueueModel:
    """Test SkeletonUpdateQueueModel functionality."""

    def test_to_dict_complete(self, db_session, project_factory):
        """Test to_dict method with all fields populated."""
        project_name = "test_to_dict_project"
        project_factory(project_name=project_name)

        now = datetime.now(timezone.utc)

        # Create model with all fields
        model = SkeletonUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=67890,
            status="processing",
            created_at=now,
            last_attempt=now,
            retry_count=2,
            error_message="Test error"
        )

        db_session.add(model)
        db_session.commit()

        # Test to_dict
        result = model.to_dict()

        # Check all fields are present and correct type
        assert result["project_name"] == project_name
        assert result["seed_id"] == 12345
        assert result["current_segment_id"] == 67890
        assert result["status"] == "processing"
        assert result["retry_count"] == 2
        assert result["error_message"] == "Test error"

        # Datetime fields are converted to ISO format strings
        assert isinstance(result["created_at"], str)
        assert isinstance(result["last_attempt"], str)
        assert result["created_at"] == now.isoformat()
        assert result["last_attempt"] == now.isoformat()

    def test_to_dict_minimal(self, db_session, project_factory):
        """Test to_dict method with minimal fields (nullables as None)."""
        project_name = "test_to_dict_minimal"
        project_factory(project_name=project_name)

        now = datetime.now(timezone.utc)

        # Create model with minimal fields
        model = SkeletonUpdateQueueModel(
            project_name=project_name,
            seed_id=12345,
            current_segment_id=None,  # nullable
            status="pending",
            created_at=now,
            last_attempt=None,  # nullable
            retry_count=0,
            error_message=None  # nullable
        )

        db_session.add(model)
        db_session.commit()

        # Test to_dict
        result = model.to_dict()

        # Check all fields are present and correct
        assert result["project_name"] == project_name
        assert result["seed_id"] == 12345
        assert result["current_segment_id"] is None
        assert result["status"] == "pending"
        assert result["retry_count"] == 0
        assert result["error_message"] is None
        assert result["last_attempt"] is None

        # created_at should be converted to ISO format string
        assert isinstance(result["created_at"], str)
        assert result["created_at"] == now.isoformat()

    def test_from_dict_complete(self, project_factory):
        """Test from_dict method with all fields populated."""
        project_name = "test_from_dict_complete"
        project_factory(project_name=project_name)

        now = datetime.now(timezone.utc)
        data = {
            "project_name": project_name,
            "seed_id": 12345,
            "current_segment_id": 67890,
            "status": "processing",
            "created_at": now.isoformat(),
            "last_attempt": now.isoformat(),
            "retry_count": 2,
            "error_message": "Test error"
        }

        # Test from_dict
        model = SkeletonUpdateQueueModel.from_dict(data)

        # Verify all fields
        assert model.project_name == project_name
        assert model.seed_id == 12345
        assert model.current_segment_id == 67890
        assert model.status == "processing"
        assert model.retry_count == 2
        assert model.error_message == "Test error"

        # Datetime fields should be parsed correctly
        assert model.created_at == now
        assert model.last_attempt == now

    def test_from_dict_minimal(self, project_factory):
        """Test from_dict method with minimal fields and defaults."""
        project_name = "test_from_dict_minimal"
        project_factory(project_name=project_name)

        now = datetime.now(timezone.utc)
        data = {
            "project_name": project_name,
            "seed_id": 12345,
            "created_at": now.isoformat(),
        }

        # Test from_dict with minimal data
        model = SkeletonUpdateQueueModel.from_dict(data)

        # Verify defaults are applied
        assert model.project_name == project_name
        assert model.seed_id == 12345
        assert model.current_segment_id is None
        assert model.status == "pending"  # default
        assert model.retry_count == 0  # default
        assert model.error_message is None
        assert model.last_attempt is None

        # created_at should be parsed from input
        assert model.created_at == now
