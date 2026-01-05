"""
Create trace_v0 tasks for segments based on various criteria.

This script provides a flexible interface for creating trace tasks for segments
based on different selection criteria like expected_segment_type, status, batch, etc.
"""

import sys
import time
from typing import Any

import click
from sqlalchemy import and_

from zetta_utils import log
from zetta_utils.task_management.db.models import SegmentModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.task_types.trace_v0 import create_trace_v0_task

logger = log.get_logger()


def create_trace_tasks(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    project_name: str,
    expected_segment_type: str | None = None,
    segment_status: str | None = None,
    segment_batch: str | None = None,
    seed_ids: list[int] | None = None,
    skip_existing_tasks: bool = True,
    task_batch_id: str = "trace_tasks",
    priority: int = 50,
    is_paused: bool = False,
    dry_run: bool = False,
    db_session: Any = None,
) -> dict[str, Any]:
    """
    Create trace_v0 tasks for segments based on selection criteria.

    Args:
        project_name: Name of the project
        expected_segment_type: Filter segments by expected_segment_type
        segment_status: Filter segments by status (Raw, Proofread, etc.)
        segment_batch: Filter segments by batch name
        seed_ids: List of specific seed_ids to create tasks for
        skip_existing_tasks: Skip segments that already have tasks
        task_batch_id: Batch ID for created tasks
        priority: Priority for created tasks (default: 50)
        is_paused: Create tasks in paused state
        dry_run: Only show what would be created
        db_session: Optional database session

    Returns:
        Dictionary with creation results
    """
    results: dict[str, Any] = {
        "segments_found": 0,
        "segments_skipped": 0,
        "tasks_created": 0,
        "creation_errors": 0,
        "created_task_ids": [],
        "errors": [],
    }

    with get_session_context(db_session) as session:
        # Build query based on criteria
        query = session.query(SegmentModel).filter(SegmentModel.project_name == project_name)

        # Apply filters
        conditions = []

        if expected_segment_type:
            conditions.append(SegmentModel.expected_segment_type == expected_segment_type)

        if segment_status:
            conditions.append(SegmentModel.status == segment_status)

        if segment_batch:
            conditions.append(SegmentModel.batch == segment_batch)

        if seed_ids:
            conditions.append(SegmentModel.seed_id.in_(seed_ids))

        # Apply all conditions
        if conditions:
            query = query.filter(and_(*conditions))

        segments = query.all()
        results["segments_found"] = len(segments)

        logger.info(f"Found {len(segments)} segments matching criteria")

        if dry_run:
            logger.info("DRY RUN MODE - No tasks will be created")
            logger.info("Selection criteria:")
            logger.info(f"  Project: {project_name}")
            logger.info(f"  Expected type: {expected_segment_type or 'Any'}")
            logger.info(f"  Status: {segment_status or 'Any'}")
            logger.info(f"  Batch: {segment_batch or 'Any'}")
            logger.info(f"  Specific seed_ids: {len(seed_ids) if seed_ids else 0}")
            logger.info(f"  Skip existing tasks: {skip_existing_tasks}")

            if segments:
                logger.info("First 5 segments that would get tasks:")
                for i, segment in enumerate(segments[:5]):
                    existing_tasks = len(segment.task_ids) if segment.task_ids else 0
                    would_skip = skip_existing_tasks and existing_tasks > 0
                    status_msg = "SKIP (has tasks)" if would_skip else "CREATE"
                    logger.info(
                        f"  {i+1}. Seed {segment.seed_id} - {segment.expected_segment_type} "
                        f"({segment.status}) - {status_msg}"
                    )

            return results

        # Create tasks
        start_time = time.time()

        for i, segment in enumerate(segments):
            # Skip if segment already has tasks and skip_existing_tasks is True
            if skip_existing_tasks and segment.task_ids:
                logger.debug(
                    f"Skipping segment {segment.seed_id} "
                    f"(already has {len(segment.task_ids)} tasks)"
                )
                results["segments_skipped"] += 1
                continue

            try:
                # Create trace_v0 task
                task_id = create_trace_v0_task(
                    project_name=project_name,
                    segment=segment,
                    kwargs={
                        "priority": priority,
                        "batch_id": task_batch_id,
                        "is_paused": is_paused,
                    },
                )

                results["tasks_created"] += 1
                results["created_task_ids"].append(task_id)

                logger.debug(f"Created task {task_id} for segment {segment.seed_id}")

                # Progress logging
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = results["tasks_created"] / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {results['tasks_created']} tasks created "
                        f"({i + 1}/{len(segments)} segments processed) - "
                        f"{rate:.1f} tasks/sec"
                    )

            except Exception as e:  # pylint: disable=broad-exception-caught
                error_msg = f"Failed to create task for segment {segment.seed_id}: {e}"
                logger.error(error_msg)
                results["creation_errors"] += 1
                results["errors"].append(error_msg)

                # Stop if too many failures
                if results["creation_errors"] > 20:
                    logger.error("Too many failures, stopping task creation")
                    break

        elapsed = time.time() - start_time

        # Log final results
        logger.info("Task creation completed:")
        logger.info(f"  Segments found: {results['segments_found']}")
        logger.info(f"  Segments skipped: {results['segments_skipped']}")
        logger.info(f"  Tasks created: {results['tasks_created']}")
        logger.info(f"  Creation errors: {results['creation_errors']}")
        logger.info(f"  Total time: {elapsed:.1f} seconds")

        if results["tasks_created"] > 0:
            logger.info(f"  Average rate: {results['tasks_created']/elapsed:.1f} tasks/sec")

    return results


@click.command()
@click.argument("project_name")
@click.option("--expected-type", help="Filter segments by expected_segment_type")
@click.option("--status", help="Filter segments by status (Raw, Proofread, Duplicate, Wrong type)")
@click.option("--batch", help="Filter segments by batch name")
@click.option("--seed-ids", help="Comma-separated list of specific seed_ids to create tasks for")
@click.option(
    "--skip-existing/--include-existing",
    default=True,
    help="Skip segments that already have tasks (default: skip)",
)
@click.option(
    "--task-batch-id",
    default="trace_tasks",
    help="Batch ID for created tasks (default: trace_tasks)",
)
@click.option("--priority", type=int, default=50, help="Priority for created tasks (default: 50)")
@click.option("--paused", is_flag=True, help="Create tasks in paused state")
@click.option(
    "--dry-run", is_flag=True, help="Only show what would be created without creating tasks"
)
def main(
    project_name: str,
    expected_type: str | None,
    status: str | None,
    batch: str | None,
    seed_ids: str | None,
    skip_existing: bool,
    task_batch_id: str,
    priority: int,
    paused: bool,
    dry_run: bool,
):
    """Create trace_v0 tasks for segments based on selection criteria.

    Examples:
        # Create tasks for all lateral segments
        python create_trace_tasks.py kronauer_ant_x0 \
            --expected-type "Olfactory Projection Right Lateral"

        # Create tasks for specific seed IDs
        python create_trace_tasks.py kronauer_ant_x0 --seed-ids "123,456,789"

        # Create tasks for segments in a specific batch
        python create_trace_tasks.py kronauer_ant_x0 \
            --batch "lateral_seed_mask_2024_08_11"

        # Dry run to see what would be created
        python create_trace_tasks.py kronauer_ant_x0 \
            --expected-type "Some Type" --dry-run
    """
    # Parse seed_ids if provided
    parsed_seed_ids = None
    if seed_ids:
        try:
            parsed_seed_ids = [int(sid.strip()) for sid in seed_ids.split(",")]
        except ValueError as e:
            logger.error(f"Invalid seed_ids format: {e}")
            sys.exit(1)

    # Validate arguments
    if not any([expected_type, status, batch, parsed_seed_ids]):
        logger.error(
            "Must specify at least one selection criteria "
            "(--expected-type, --status, --batch, or --seed-ids)"
        )
        sys.exit(1)

    if not dry_run:
        # Confirm before creating tasks
        criteria_parts = []
        if expected_type:
            criteria_parts.append(f"expected_type='{expected_type}'")
        if status:
            criteria_parts.append(f"status='{status}'")
        if batch:
            criteria_parts.append(f"batch='{batch}'")
        if parsed_seed_ids:
            criteria_parts.append(f"{len(parsed_seed_ids)} specific seed_ids")

        criteria_str = ", ".join(criteria_parts)

        print(f"\nThis will create trace_v0 tasks for segments matching: {criteria_str}")
        print(f"Project: {project_name}")
        print(f"Task batch ID: {task_batch_id}")
        print(f"Priority: {priority}")
        print(f"Skip existing tasks: {skip_existing}")
        print(f"Paused: {paused}")

        response = input("\nContinue? (y/N): ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    # Create tasks
    results = create_trace_tasks(
        project_name=project_name,
        expected_segment_type=expected_type,
        segment_status=status,
        segment_batch=batch,
        seed_ids=parsed_seed_ids,
        skip_existing_tasks=skip_existing,
        task_batch_id=task_batch_id,
        priority=priority,
        is_paused=paused,
        dry_run=dry_run,
    )

    if not dry_run and results["tasks_created"] > 0:
        print(f"\n✅ Successfully created {results['tasks_created']} trace_v0 tasks")
        if results["creation_errors"] > 0:
            print(f"⚠️  {results['creation_errors']} errors occurred")
