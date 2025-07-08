"""
Trace Postprocessor Worker

An automated worker that polls for trace_postprocess_v0 tasks and updates segment statistics
(skeleton length and synapse counts) from CAVE.
"""

import random
import time

import click

from zetta_utils import log
from zetta_utils.task_management.db.models import TaskModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment import update_segment_statistics
from zetta_utils.task_management.task import (
    create_task,
    get_task,
    release_task,
    start_task,
)
from zetta_utils.task_management.types import Task
from zetta_utils.task_management.utils import generate_id_nonunique

logger = log.get_logger()


def process_task(task_id: str, project_name: str, user_id: str, task_count: int) -> None:
    """Process a single trace_postprocess_v0 task."""
    try:
        logger.info(f"Processing task {task_id} (#{task_count})")

        # Get task details
        task_details = get_task(project_name=project_name, task_id=task_id)
        extra_data = task_details.get("extra_data")
        assert extra_data is not None, "Task extra_data is None"
        original_task = get_task(project_name=project_name, task_id=extra_data["original_task_id"])
        seed_id = extra_data["seed_id"]
        original_user = original_task["completed_user_id"]

        logger.info(
            f"Original task: {original_task['task_id']}, user: {original_user}, seed: {seed_id}"
        )

        # Update segment statistics
        update_segment_statistics(project_name=project_name, seed_id=seed_id)
        logger.info(f"Updated statistics for segment {seed_id}")

        # Create feedback task based on user
        if original_user == "sergiy@zetta.ai":
            create_feedback_task_from_trace(project_name, original_task["task_id"])
            logger.info("Created feedback task (100% for sergiy@zetta.ai)")
        elif random.random() < 0.1:
            create_feedback_task_from_trace(project_name, original_task["task_id"])
            logger.info("Created feedback task (10% sample)")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(e)

    finally:
        # Always release the task
        release_task(
            project_name=project_name, task_id=task_id, user_id=user_id, completion_status="Done"
        )
        logger.info(f"Released task {task_id}")


@click.command()
@click.option(
    "--user_id",
    "-u",
    default="automated_worker",
    help="User ID for the automated worker (default: automated_worker)",
)
@click.option(
    "--project_name", "-p", required=True, help="Name of the project to process tasks from"
)
@click.option(
    "--polling_period",
    "-t",
    type=float,
    default=5.0,
    help="Polling period in seconds (default: 5.0)",
)
def run_worker(user_id: str, project_name: str, polling_period: float):
    """Run the trace postprocessor worker."""
    logger.info(
        f"Starting trace postprocessor worker for project '{project_name}' with user '{user_id}'"
    )

    task_count = 0

    try:
        while True:
            try:
                task_id = start_task(project_name=project_name, user_id=user_id)

                if task_id:
                    task_count += 1
                    logger.info(f"Started processing task {task_id}")
                    process_task(task_id, project_name, user_id, task_count)
                else:
                    time.sleep(polling_period)

            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Error in worker loop")
                time.sleep(60)  # Wait 1 minute before retrying

    except KeyboardInterrupt:
        logger.info(f"Worker stopped by user after processing {task_count} tasks")


def create_feedback_task_from_trace(project_name: str, trace_task_id: str) -> None:
    """Create a trace_feedback_v0 task for a completed trace task.

    Args:
        project_name: The project name
        trace_task_id: The ID of the completed trace task
    """
    with get_session_context() as session:
        # Get the trace task
        trace_task = (
            session.query(TaskModel)
            .filter(
                TaskModel.project_name == project_name,
                TaskModel.task_id == trace_task_id,
            )
            .first()
        )

        if not trace_task:
            logger.warning(f"Trace task {trace_task_id} not found")
            return

        # Generate feedback task ID
        feedback_task_id = f"feedback_{trace_task_id}_{generate_id_nonunique()}"

        # Create feedback task with the trace task's final ng_state
        feedback_task = Task(
            task_id=feedback_task_id,
            task_type="trace_feedback_v0",
            ng_state=trace_task.ng_state,  # Use the completed trace's state
            ng_state_initial=trace_task.ng_state,  # Same for initial state
            completion_status="",
            assigned_user_id="",
            active_user_id="",
            completed_user_id="",
            priority=70,  # High priority for feedback
            batch_id="feedback",
            last_leased_ts=0.0,
            is_active=True,
            is_paused=False,
            is_checked=False,
            extra_data={
                "original_task_id": trace_task.task_id,
            },
        )

        # Create the task
        created_task_id = create_task(
            project_name=project_name, data=feedback_task, db_session=session
        )

        logger.info(f"Created feedback task {created_task_id} for trace task {trace_task_id}")


if __name__ == "__main__":
    run_worker()  # pylint: disable=no-value-for-parameter
