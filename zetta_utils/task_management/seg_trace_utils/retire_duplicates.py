#!/usr/bin/env python3
"""
Task-completion based retirement utility for duplicate segments.

This utility finds segments with duplicate current_segment_id values and implements
a sophisticated retirement algorithm that keeps the segment with the most recently
completed task, retiring all others in each duplicate group.
"""

from datetime import datetime

from sqlalchemy import text

from zetta_utils import log
from zetta_utils.task_management.db.models import SegmentModel, TaskModel
from zetta_utils.task_management.db.session import get_session_context

logger = log.get_logger()


def find_duplicate_current_segment_ids(
    project_name: str, dry_run: bool = True  # pylint: disable=unused-argument
) -> dict[int, list[dict]]:
    """
    Find all non-Duplicate segments with duplicate current_segment_id values.

    Args:
        project_name: The project to search in
        dry_run: If True, only log what would be done without making changes

    Returns:
        Dictionary mapping current_segment_id to list of segment info dictionaries
    """
    logger.info(f"Finding duplicate current_segment_id values in project '{project_name}'")

    with get_session_context() as session:
        # Find all current_segment_id values that have duplicates among non-Duplicate segments
        duplicate_query = text(
            """
            SELECT current_segment_id, COUNT(*) as duplicate_count
            FROM segments
            WHERE project_name = :project_name
            AND current_segment_id IS NOT NULL
            AND status != 'Duplicate'
            GROUP BY current_segment_id
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC, current_segment_id
        """
        )

        duplicate_results = session.execute(
            duplicate_query, {"project_name": project_name}
        ).fetchall()

        if not duplicate_results:
            logger.info("No duplicate current_segment_id values found")
            return {}

        logger.info(f"Found {len(duplicate_results)} current_segment_id values with duplicates")

        # Get detailed information for all segments in duplicate groups
        duplicates = {}

        for row in duplicate_results:
            current_segment_id = row.current_segment_id
            duplicate_count = row.duplicate_count

            logger.info(
                f"Processing current_segment_id {current_segment_id} "
                f"with {duplicate_count} duplicates"
            )

            # Get all segments with this current_segment_id
            segments = (
                session.query(SegmentModel)
                .filter(
                    SegmentModel.project_name == project_name,
                    SegmentModel.current_segment_id == current_segment_id,
                    SegmentModel.status != "Duplicate",
                )
                .all()
            )

            segment_info = []
            for segment in segments:
                # Get the most recent completed task for this segment
                most_recent_task = None
                if segment.task_ids:
                    tasks = (
                        session.query(TaskModel)
                        .filter(
                            TaskModel.project_name == project_name,
                            TaskModel.task_id.in_(segment.task_ids),
                            TaskModel.completion_status != "",
                        )
                        .order_by(TaskModel.last_leased_ts.desc())
                        .first()
                    )
                    most_recent_task = tasks

                segment_info.append(
                    {
                        "seed_id": segment.seed_id,
                        "status": segment.status,
                        "expected_segment_type": segment.expected_segment_type,
                        "segment_type": segment.segment_type,
                        "task_ids": segment.task_ids,
                        "updated_at": segment.updated_at,
                        "most_recent_task": most_recent_task,
                        "most_recent_task_ts": (
                            most_recent_task.last_leased_ts if most_recent_task else 0.0
                        ),
                    }
                )

            duplicates[current_segment_id] = segment_info

            # Log details for this duplicate group
            logger.info(f"  current_segment_id {current_segment_id} duplicates:")
            for info in segment_info:
                task_info = (
                    f"task_ts={info['most_recent_task_ts']}"
                    if info["most_recent_task"]
                    else "no_completed_tasks"
                )
                logger.info(
                    f"    seed_id={info['seed_id']}, status={info['status']}, "
                    f"type={info['expected_segment_type']}, {task_info}"
                )

        return duplicates


def retire_duplicate_segments(project_name: str, dry_run: bool = True) -> dict[str, int]:
    """
    Retire duplicate segments using task-completion based algorithm.

    Algorithm:
    1. Find all current_segment_id values with duplicates among non-Duplicate segments
    2. For each duplicate group:
       - If segments have different expected_segment_type, retire "undetermined" types first
       - Among remaining segments, keep the one with most recently completed task
         (highest last_leased_ts)
       - If no completed tasks exist, keep the most recently updated segment
       - Retire all other segments in the group

    Args:
        project_name: The project to process
        dry_run: If True, only log what would be done without making changes

    Returns:
        Dictionary with retirement statistics
    """
    logger.info(f"Starting duplicate segment retirement for project '{project_name}'")
    logger.info(f"Dry run mode: {dry_run}")

    # Find duplicate groups
    duplicates = find_duplicate_current_segment_ids(project_name, dry_run)

    if not duplicates:
        logger.info("No duplicates found, nothing to retire")
        return {"duplicate_groups": 0, "segments_retired": 0, "segments_kept": 0}

    duplicate_stats = {
        "duplicate_groups": len(duplicates),
        "segments_retired": 0,
        "segments_kept": 0,
    }

    with get_session_context() as session:
        for current_segment_id, segment_infos in duplicates.items():
            logger.info(
                f"\nProcessing duplicate group for current_segment_id {current_segment_id}"
            )

            # Apply retirement algorithm
            winner_info = _select_winner_segment(segment_infos)

            logger.info(
                f"Selected winner: seed_id={winner_info['seed_id']}, "
                f"type={winner_info['expected_segment_type']}, "
                f"task_ts={winner_info['most_recent_task_ts']}"
            )

            # Retire all other segments in this group
            for info in segment_infos:
                if info["seed_id"] == winner_info["seed_id"]:
                    duplicate_stats["segments_kept"] += 1
                    logger.info(f"  Keeping segment seed_id={info['seed_id']}")
                else:
                    duplicate_stats["segments_retired"] += 1
                    logger.info(f"  Retiring segment seed_id={info['seed_id']}")

                    if not dry_run:
                        # Update segment status to Duplicate
                        segment = (
                            session.query(SegmentModel)
                            .filter(
                                SegmentModel.project_name == project_name,
                                SegmentModel.seed_id == info["seed_id"],
                            )
                            .first()
                        )

                        if segment:
                            segment.status = "Duplicate"
                            segment.updated_at = datetime.now()
                            logger.info(
                                f"    Updated segment seed_id={info['seed_id']} "
                                "status to Duplicate"
                            )

        if not dry_run:
            session.commit()
            logger.info("Committed all retirement updates to database")
        else:
            logger.info("Dry run complete - no changes made to database")

    logger.info("\nRetirement summary:")
    logger.info(f"  Duplicate groups processed: {duplicate_stats['duplicate_groups']}")
    logger.info(f"  Segments retired: {duplicate_stats['segments_retired']}")
    logger.info(f"  Segments kept: {duplicate_stats['segments_kept']}")

    return duplicate_stats


def _select_winner_segment(segment_infos: list[dict]) -> dict:
    """
    Select the winner segment from a duplicate group using business logic priority.

    Priority rules:
    1. Segments with determined types (not "undetermined") win over undetermined
    2. Among segments of same type preference, most recent completed task wins
    3. If no completed tasks, most recently updated segment wins

    Args:
        segment_infos: List of segment info dictionaries

    Returns:
        The winning segment info dictionary
    """
    logger.info(f"Selecting winner from {len(segment_infos)} duplicate segments")

    # Separate segments by type determination
    determined_segments = [
        s for s in segment_infos if s["expected_segment_type"] != "undetermined"
    ]
    undetermined_segments = [
        s for s in segment_infos if s["expected_segment_type"] == "undetermined"
    ]

    # Prefer determined types over undetermined
    candidates = determined_segments if determined_segments else undetermined_segments

    logger.info(
        f"  Determined segments: {len(determined_segments)}, "
        f"Undetermined: {len(undetermined_segments)}"
    )
    candidate_type = "determined" if determined_segments else "undetermined"
    logger.info(f"  Using candidates from: {candidate_type} group")

    # Sort by task completion timestamp (most recent first), then by updated_at
    def sort_key(segment_info):
        task_ts = segment_info["most_recent_task_ts"]
        updated_at = segment_info["updated_at"].timestamp() if segment_info["updated_at"] else 0.0
        return (-task_ts, -updated_at)  # Negative for descending order

    candidates.sort(key=sort_key)

    winner = candidates[0]

    logger.info("  Winner selection criteria:")
    logger.info(f"    Task timestamp: {winner['most_recent_task_ts']}")
    logger.info(f"    Updated at: {winner['updated_at']}")
    logger.info(f"    Type: {winner['expected_segment_type']}")

    return winner


def retire_specific_segments(
    specific_project_name: str, current_segment_ids: list[int], dry_run: bool = True
) -> dict[str, int]:
    """
    Retire duplicate segments for specific current_segment_id values only.

    Args:
        project_name: The project to process
        current_segment_ids: List of specific current_segment_id values to process
        dry_run: If True, only log what would be done without making changes

    Returns:
        Dictionary with retirement statistics
    """
    logger.info(f"Processing specific current_segment_ids: {current_segment_ids}")

    # Get all duplicates first
    all_duplicates = find_duplicate_current_segment_ids(specific_project_name, dry_run)

    # Filter to only the requested IDs
    filtered_duplicates = {
        seg_id: infos for seg_id, infos in all_duplicates.items() if seg_id in current_segment_ids
    }

    if not filtered_duplicates:
        logger.info("No duplicates found for the specified current_segment_ids")
        return {"duplicate_groups": 0, "segments_retired": 0, "segments_kept": 0}

    logger.info(f"Found {len(filtered_duplicates)} duplicate groups to process")

    # Process using the same algorithm
    specific_stats = {
        "duplicate_groups": len(filtered_duplicates),
        "segments_retired": 0,
        "segments_kept": 0,
    }

    with get_session_context() as session:
        for current_segment_id, segment_infos in filtered_duplicates.items():
            logger.info(
                f"\nProcessing duplicate group for current_segment_id {current_segment_id}"
            )

            winner_info = _select_winner_segment(segment_infos)

            logger.info(f"Selected winner: seed_id={winner_info['seed_id']}")

            for info in segment_infos:
                if info["seed_id"] == winner_info["seed_id"]:
                    specific_stats["segments_kept"] += 1
                    logger.info(f"  Keeping segment seed_id={info['seed_id']}")
                else:
                    specific_stats["segments_retired"] += 1
                    logger.info(f"  Retiring segment seed_id={info['seed_id']}")

                    if not dry_run:
                        segment = (
                            session.query(SegmentModel)
                            .filter(
                                SegmentModel.project_name == specific_project_name,
                                SegmentModel.seed_id == info["seed_id"],
                            )
                            .first()
                        )

                        if segment:
                            segment.status = "Duplicate"
                            segment.updated_at = datetime.now()

        if not dry_run:
            session.commit()
            logger.info("Committed specific retirement updates to database")

    return specific_stats


if __name__ == "__main__":
    # Default to dry run for safety
    main_project_name = "kronauer_ant_x0"

    print("Task-completion based duplicate segment retirement utility")
    print("=" * 60)
    print(f"Project: {main_project_name}")
    print("Mode: DRY RUN (no changes will be made)")
    print()

    # Run the retirement algorithm
    final_stats = retire_duplicate_segments(main_project_name, dry_run=True)

    print("\nSummary:")
    print(f"Duplicate groups found: {final_stats['duplicate_groups']}")
    print(f"Segments that would be retired: {final_stats['segments_retired']}")
    print(f"Segments that would be kept: {final_stats['segments_kept']}")
    print()
    print("To actually perform the retirement, modify dry_run=False in the script")
