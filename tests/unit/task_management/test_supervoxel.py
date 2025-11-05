"""Tests for supervoxel module"""

# pylint: disable=unused-argument,redefined-outer-name

from datetime import datetime, timezone

import pytest

from zetta_utils.task_management.db.models import SegmentEditEventModel, SupervoxelModel
from zetta_utils.task_management.supervoxel import (
    create_supervoxel,
    get_supervoxels_by_segment,
    update_supervoxels_for_merge,
    update_supervoxels_for_split,
)


@pytest.fixture
def existing_supervoxels(clean_db, db_session, project_name):
    """Create supervoxels in the database"""
    supervoxels = [
        SupervoxelModel(
            supervoxel_id=100,
            seed_x=10.0,
            seed_y=20.0,
            seed_z=30.0,
            current_segment_id=1000,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        SupervoxelModel(
            supervoxel_id=101,
            seed_x=15.0,
            seed_y=25.0,
            seed_z=35.0,
            current_segment_id=1000,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        SupervoxelModel(
            supervoxel_id=102,
            seed_x=20.0,
            seed_y=30.0,
            seed_z=40.0,
            current_segment_id=2000,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    ]
    db_session.add_all(supervoxels)
    db_session.commit()
    yield supervoxels


def test_create_supervoxel(clean_db, db_session):
    """Test creating a supervoxel"""
    supervoxel_id = 123
    seed_x = 100.0
    seed_y = 200.0
    seed_z = 300.0
    segment_id = 456

    create_supervoxel(
        supervoxel_id=supervoxel_id,
        seed_x=seed_x,
        seed_y=seed_y,
        seed_z=seed_z,
        current_segment_id=segment_id,
        db_session=db_session,
    )

    # Verify supervoxel was created
    db_session.expire_all()  # Refresh session
    supervoxel = (
        db_session.query(SupervoxelModel)
        .filter_by(supervoxel_id=supervoxel_id)
        .first()
    )
    assert supervoxel is not None
    assert supervoxel.seed_x == seed_x
    assert supervoxel.seed_y == seed_y
    assert supervoxel.seed_z == seed_z
    assert supervoxel.current_segment_id == segment_id


def test_get_supervoxels_by_segment(existing_supervoxels, db_session):
    """Test getting supervoxels by segment ID"""
    # Get supervoxels for segment 1000
    supervoxels = get_supervoxels_by_segment(segment_id=1000, db_session=db_session)

    assert len(supervoxels) == 2
    assert all(sv.current_segment_id == 1000 for sv in supervoxels)
    assert {sv.supervoxel_id for sv in supervoxels} == {100, 101}

    # Get supervoxels for segment 2000
    supervoxels = get_supervoxels_by_segment(segment_id=2000, db_session=db_session)

    assert len(supervoxels) == 1
    assert supervoxels[0].current_segment_id == 2000
    assert supervoxels[0].supervoxel_id == 102


def test_update_supervoxels_for_merge(
    existing_supervoxels, db_session, project_name
):
    """Test updating supervoxels for a merge operation"""
    old_root_ids = [1000, 2000]
    new_root_id = 3000
    event_id = "merge_123"
    edit_timestamp = datetime.now(timezone.utc)

    # Perform merge
    count = update_supervoxels_for_merge(
        old_root_ids=old_root_ids,
        new_root_id=new_root_id,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="merge",
        db_session=db_session,
    )

    # Should update all 3 supervoxels
    assert count == 3

    # Verify all supervoxels now point to new root
    db_session.expire_all()  # Refresh session
    supervoxels = db_session.query(SupervoxelModel).all()
    assert len(supervoxels) == 3
    assert all(sv.current_segment_id == new_root_id for sv in supervoxels)

    # Verify merge event was recorded
    merge_event = (
        db_session.query(SegmentEditEventModel)
        .filter_by(project_name=project_name, event_id=event_id)
        .first()
    )
    assert merge_event is not None
    assert merge_event.old_root_ids == old_root_ids
    assert merge_event.new_root_ids == [new_root_id]  # Unified model uses list
    assert merge_event.operation_type == "merge"


def test_update_supervoxels_for_merge_idempotent(
    existing_supervoxels, db_session, project_name
):
    """Test that merge operations are idempotent"""
    old_root_ids = [1000]
    new_root_id = 3000
    event_id = "merge_456"
    edit_timestamp = datetime.now(timezone.utc)

    # Perform merge first time
    count1 = update_supervoxels_for_merge(
        old_root_ids=old_root_ids,
        new_root_id=new_root_id,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="merge",
        db_session=db_session,
    )
    assert count1 == 2

    # Perform same merge again (should be idempotent)
    count2 = update_supervoxels_for_merge(
        old_root_ids=old_root_ids,
        new_root_id=new_root_id,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="merge",
        db_session=db_session,
    )
    assert count2 == 0  # No updates since already processed

    # Verify only one merge event recorded
    merge_events = (
        db_session.query(SegmentEditEventModel)
        .filter_by(project_name=project_name, event_id=event_id)
        .all()
    )
    assert len(merge_events) == 1


def test_update_supervoxels_for_split(
    existing_supervoxels, db_session, project_name
):
    """Test updating supervoxels for a split operation"""
    old_root_id = 1000
    new_root_ids = [4000, 5000]
    supervoxel_assignments = {100: 4000, 101: 5000}
    event_id = "split_123"
    edit_timestamp = datetime.now(timezone.utc)

    # Verify initial state
    supervoxels = get_supervoxels_by_segment(segment_id=old_root_id, db_session=db_session)
    assert len(supervoxels) == 2
    assert all(sv.current_segment_id == old_root_id for sv in supervoxels)

    # Perform split
    count = update_supervoxels_for_split(
        old_root_id=old_root_id,
        new_root_ids=new_root_ids,
        supervoxel_assignments=supervoxel_assignments,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="split",
        db_session=db_session,
    )

    # Should update 2 supervoxels
    assert count == 2

    # Verify supervoxels were updated correctly
    db_session.expire_all()  # Refresh session
    supervoxel_100 = (
        db_session.query(SupervoxelModel)
        .filter_by(supervoxel_id=100)
        .first()
    )
    assert supervoxel_100.current_segment_id == 4000

    supervoxel_101 = (
        db_session.query(SupervoxelModel)
        .filter_by(supervoxel_id=101)
        .first()
    )
    assert supervoxel_101.current_segment_id == 5000

    # Verify other supervoxels unchanged
    supervoxel_102 = (
        db_session.query(SupervoxelModel)
        .filter_by(supervoxel_id=102)
        .first()
    )
    assert supervoxel_102.current_segment_id == 2000

    # Verify split event was recorded
    split_event = (
        db_session.query(SegmentEditEventModel)
        .filter_by(project_name=project_name, event_id=event_id)
        .first()
    )
    assert split_event is not None
    assert split_event.old_root_ids == [old_root_id]  # Unified model uses list
    assert split_event.new_root_ids == new_root_ids
    assert split_event.operation_type == "split"


def test_update_supervoxels_for_split_idempotent(
    existing_supervoxels, db_session, project_name
):
    """Test that split operations are idempotent"""
    old_root_id = 1000
    new_root_ids = [4000, 5000]
    supervoxel_assignments = {100: 4000, 101: 5000}
    event_id = "split_456"
    edit_timestamp = datetime.now(timezone.utc)

    # Perform split first time
    count1 = update_supervoxels_for_split(
        old_root_id=old_root_id,
        new_root_ids=new_root_ids,
        supervoxel_assignments=supervoxel_assignments,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="split",
        db_session=db_session,
    )
    assert count1 == 2

    # Perform same split again (should be idempotent)
    count2 = update_supervoxels_for_split(
        old_root_id=old_root_id,
        new_root_ids=new_root_ids,
        supervoxel_assignments=supervoxel_assignments,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="split",
        db_session=db_session,
    )
    assert count2 == 0  # No updates since already processed

    # Verify only one split event recorded
    split_events = (
        db_session.query(SegmentEditEventModel)
        .filter_by(project_name=project_name, event_id=event_id)
        .all()
    )
    assert len(split_events) == 1


def test_update_supervoxels_for_split_empty_assignments(
    existing_supervoxels, db_session, project_name
):
    """Test split operation with empty assignments"""
    old_root_id = 1000
    new_root_ids = [4000, 5000]
    supervoxel_assignments: dict[int, int] = {}  # Empty assignments
    event_id = "split_empty"
    edit_timestamp = datetime.now(timezone.utc)

    # Perform split with no assignments
    count = update_supervoxels_for_split(
        old_root_id=old_root_id,
        new_root_ids=new_root_ids,
        supervoxel_assignments=supervoxel_assignments,
        project_name=project_name,
        event_id=event_id,
        edit_timestamp=edit_timestamp,
        operation_type="split",
        db_session=db_session,
    )

    # Should update 0 supervoxels
    assert count == 0

    # Verify split event was still recorded
    split_event = (
        db_session.query(SegmentEditEventModel)
        .filter_by(project_name=project_name, event_id=event_id)
        .first()
    )
    assert split_event is not None


def test_get_supervoxels_by_segment_empty(clean_db, db_session):
    """Test getting supervoxels when none exist"""
    supervoxels = get_supervoxels_by_segment(segment_id=9999, db_session=db_session)
    assert len(supervoxels) == 0
