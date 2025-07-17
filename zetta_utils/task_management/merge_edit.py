from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from .db.models import MergeEditModel
from .db.session import get_session_context


def create_merge_edit(
    project_name: str,
    task_id: str,
    user_id: str,
    points: List[List],
    db_session: Optional[Session] = None,
) -> int:
    """
    Create a new merge edit record.

    Args:
        project_name: Name of the project
        task_id: ID of the task this edit belongs to
        user_id: ID of the user who made this edit
        points: List of two points to merge [segment_id, x, y, z]
        db_session: Optional database session

    Returns:
        edit_id: ID of the created merge edit
    """
    with get_session_context(db_session) as session:
        merge_edit = MergeEditModel(
            project_name=project_name,
            task_id=task_id,
            user_id=user_id,
            points=points,
            created_at=datetime.now(),
        )

        session.add(merge_edit)
        session.commit()
        session.refresh(merge_edit)

        return merge_edit.edit_id


def get_merge_edits_by_task(
    project_name: str,
    task_id: str,
    db_session: Optional[Session] = None,
) -> List[Dict]:
    """
    Get all merge edits for a specific task.

    Args:
        project_name: Name of the project
        task_id: ID of the task
        db_session: Optional database session

    Returns:
        List of merge edit dictionaries
    """
    with get_session_context(db_session) as session:
        merge_edits = (
            session.query(MergeEditModel)
            .filter(
                MergeEditModel.project_name == project_name,
                MergeEditModel.task_id == task_id,
            )
            .order_by(MergeEditModel.created_at.desc())
            .all()
        )

        return [edit.to_dict() for edit in merge_edits]


def get_merge_edits_by_user(
    project_name: str,
    user_id: str,
    db_session: Optional[Session] = None,
) -> List[Dict]:
    """
    Get all merge edits for a specific user.

    Args:
        project_name: Name of the project
        user_id: ID of the user
        db_session: Optional database session

    Returns:
        List of merge edit dictionaries
    """
    with get_session_context(db_session) as session:
        merge_edits = (
            session.query(MergeEditModel)
            .filter(
                MergeEditModel.project_name == project_name,
                MergeEditModel.user_id == user_id,
            )
            .order_by(MergeEditModel.created_at.desc())
            .all()
        )

        return [edit.to_dict() for edit in merge_edits]


def get_merge_edit_by_id(
    project_name: str,
    edit_id: int,
    db_session: Optional[Session] = None,
) -> Optional[Dict]:
    """
    Get a specific merge edit by its ID.

    Args:
        project_name: Name of the project
        edit_id: ID of the merge edit
        db_session: Optional database session

    Returns:
        Merge edit dictionary or None if not found
    """
    with get_session_context(db_session) as session:
        merge_edit = (
            session.query(MergeEditModel)
            .filter(
                MergeEditModel.project_name == project_name,
                MergeEditModel.edit_id == edit_id,
            )
            .first()
        )

        return merge_edit.to_dict() if merge_edit else None
