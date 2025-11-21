"""Tests for SkeletonUpdateQueueModel."""

from datetime import datetime, timezone

import pytest

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