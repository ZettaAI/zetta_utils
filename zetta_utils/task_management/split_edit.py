from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from .db.models import SplitEditModel
from .db.session import get_session_context


def create_split_edit(
    project_name: str,
    task_id: str,
    user_id: str,
    sources: List[List],
    sinks: List[List],
    db_session: Optional[Session] = None,
) -> int:
    """
    Create a new split edit record.
    
    Args:
        project_name: Name of the project
        task_id: ID of the task this edit belongs to
        user_id: ID of the user who made this edit
        sources: List of source points [segment_id, x, y, z]
        sinks: List of sink points [segment_id, x, y, z]
        db_session: Optional database session
        
    Returns:
        edit_id: ID of the created split edit
    """
    with get_session_context(db_session) as session:
        split_edit = SplitEditModel(
            project_name=project_name,
            task_id=task_id,
            user_id=user_id,
            sources=sources,
            sinks=sinks,
            created_at=datetime.now(),
        )
        
        session.add(split_edit)
        session.commit()
        session.refresh(split_edit)
        
        return split_edit.edit_id


def get_split_edits_by_task(
    project_name: str,
    task_id: str,
    db_session: Optional[Session] = None,
) -> List[Dict]:
    """
    Get all split edits for a specific task.
    
    Args:
        project_name: Name of the project
        task_id: ID of the task
        db_session: Optional database session
        
    Returns:
        List of split edit dictionaries
    """
    with get_session_context(db_session) as session:
        split_edits = (
            session.query(SplitEditModel)
            .filter(
                SplitEditModel.project_name == project_name,
                SplitEditModel.task_id == task_id,
            )
            .order_by(SplitEditModel.created_at.desc())
            .all()
        )
        
        return [edit.to_dict() for edit in split_edits]


def get_split_edits_by_user(
    project_name: str,
    user_id: str,
    db_session: Optional[Session] = None,
) -> List[Dict]:
    """
    Get all split edits for a specific user.
    
    Args:
        project_name: Name of the project
        user_id: ID of the user
        db_session: Optional database session
        
    Returns:
        List of split edit dictionaries
    """
    with get_session_context(db_session) as session:
        split_edits = (
            session.query(SplitEditModel)
            .filter(
                SplitEditModel.project_name == project_name,
                SplitEditModel.user_id == user_id,
            )
            .order_by(SplitEditModel.created_at.desc())
            .all()
        )
        
        return [edit.to_dict() for edit in split_edits]


def get_split_edit_by_id(
    project_name: str,
    edit_id: int,
    db_session: Optional[Session] = None,
) -> Optional[Dict]:
    """
    Get a specific split edit by its ID.
    
    Args:
        project_name: Name of the project
        edit_id: ID of the split edit
        db_session: Optional database session
        
    Returns:
        Split edit dictionary or None if not found
    """
    with get_session_context(db_session) as session:
        split_edit = (
            session.query(SplitEditModel)
            .filter(
                SplitEditModel.project_name == project_name,
                SplitEditModel.edit_id == edit_id,
            )
            .first()
        )
        
        return split_edit.to_dict() if split_edit else None