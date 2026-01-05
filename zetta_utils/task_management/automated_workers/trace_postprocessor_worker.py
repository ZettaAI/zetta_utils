"""
Trace Postprocessor Worker

An automated worker that polls for trace_postprocess_v0 tasks and updates segment statistics
(skeleton length and synapse counts) from CAVE.
"""

import time

import click

from zetta_utils import log
from zetta_utils.task_management.db.models import SegmentModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment import update_segment_statistics
from zetta_utils.task_management.task import get_task, release_task, start_task

logger = log.get_logger()


def process_task(
    task_id: str,
    project_name: str,
    user_id: str,
    task_count: int,
    expected_segment_type: str | None = None,
) -> None:
    """Process a single trace_postprocess_v0 task."""
    try:
        logger.info(f"Processing task {task_id} (#{task_count})")

        # Get task details
        task_details = get_task(project_name=project_name, task_id=task_id)
        extra_data = task_details.get("extra_data")
        assert extra_data is not None, "Task extra_data is None"
        original_task = get_task(project_name=project_name, task_id=extra_data["original_task_id"])
        original_task_extra_data = original_task.get("extra_data")
        assert original_task_extra_data is not None, "Original task extra_data is None"
        seed_id = original_task_extra_data["seed_id"]
        original_user = original_task["completed_user_id"]

        logger.info(
            f"Original task: {original_task['task_id']}, user: {original_user}, seed: {seed_id}"
        )

        # Check segment type filter if specified
        if expected_segment_type:
            with get_session_context() as session:
                segment = (
                    session.query(SegmentModel)
                    .filter_by(project_name=project_name, seed_id=seed_id)
                    .first()
                )

                if not segment or segment.expected_segment_type != expected_segment_type:
                    logger.info(
                        f"Skipping segment {seed_id} - expected_segment_type "
                        f"'{segment.expected_segment_type if segment else 'None'}' does not match "
                        f"filter '{expected_segment_type}'"
                    )
                    return

        # Update segment statistics
        update_segment_statistics(project_name=project_name, seed_id=seed_id)
        logger.info(f"Updated statistics for segment {seed_id}")

        # Create feedback task based on user
        # if original_user == "sergiy@zetta.ai":
        #    feedback_id = create_feedback_task_from_trace(
        #        project_name=project_name,
        #        trace_task_id=original_task["task_id"],
        #        add_status_annotation=True,
        #    )
        #    if feedback_id:
        #        logger.info(f"Created feedback task {feedback_id} (100% for sergiy@zetta.ai)")
        # elif random.random() < 0.1:
        #    feedback_id = create_feedback_task_from_trace(
        #        project_name=project_name,
        #        trace_task_id=original_task["task_id"],
        #        add_status_annotation=True,
        #    )
        #    if feedback_id:
        #        logger.info(f"Created feedback task {feedback_id} (10% sample)")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(e)

    finally:
        # Always release the task
        release_task(
            project_name=project_name,
            task_id=task_id,
            user_id=user_id,
            completion_status="Proofread",
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
@click.option(
    "--expected-segment-type",
    "-e",
    type=str,
    default=None,
    help="Only process tasks for segments with this expected_segment_type",
)
def run_worker(
    user_id: str, project_name: str, polling_period: float, expected_segment_type: str | None
):
    """Run the trace postprocessor worker."""
    filter_info = (
        f" (filtering by expected_segment_type: {expected_segment_type})"
        if expected_segment_type
        else ""
    )
    logger.info(
        f"Starting trace postprocessor worker for project '{project_name}' "
        f"with user '{user_id}'{filter_info}"
    )

    task_count = 0

    try:
        while True:
            try:
                task_id = start_task(project_name=project_name, user_id=user_id)

                if task_id:
                    task_count += 1
                    logger.info(f"Started processing task {task_id}")
                    process_task(task_id, project_name, user_id, task_count, expected_segment_type)
                else:
                    time.sleep(polling_period)

            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Error in worker loop")
                time.sleep(60)  # Wait 1 minute before retrying

    except KeyboardInterrupt:
        logger.info(f"Worker stopped by user after processing {task_count} tasks")


if __name__ == "__main__":
    run_worker()  # pylint: disable=no-value-for-parameter
