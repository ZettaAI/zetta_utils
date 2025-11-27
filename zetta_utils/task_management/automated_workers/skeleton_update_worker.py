"""
Skeleton Update Worker

Background worker that processes skeleton length updates from the queue.
Calls Cave API to get updated skeleton lengths after PCG edits.
"""

import time

import click

from zetta_utils import log
from zetta_utils.task_management.segment import update_segment_info
from zetta_utils.task_management.skeleton_queue import (
    cleanup_completed_updates,
    get_next_pending_update,
    get_queue_stats,
    mark_update_completed,
    mark_update_failed,
    mark_update_processing,
)

logger = log.get_logger()


def process_skeleton_update(
    project_name: str,
    seed_id: int,
    server_address: str = "https://proofreading.zetta.ai",
) -> bool:
    """
    Process a single skeleton update by calling Cave API.
    
    Args:
        project_name: Project name
        seed_id: Seed ID to update
        server_address: Cave server address
        
    Returns:
        True if successful, False if failed
    """
    try:
        print(f"[DEBUG] Processing skeleton update for seed {seed_id}")
        logger.info(f"Processing skeleton update for seed {seed_id}")

        print(
            f"[DEBUG] Calling update_segment_info with project={project_name}, "  # pylint: disable=line-too-long
            f"seed={seed_id}, server={server_address}"  # pylint: disable=line-too-long
        )  # pylint: disable=line-too-long
        # Update segment statistics including skeleton length
        # This function gets the current segment ID and calls Cave API
        result = update_segment_info(
            project_name=project_name,
            seed_id=seed_id,
            server_address=server_address
        )

        print(f"[DEBUG] update_segment_info result: {result}")

        if "error" in result:
            print(f"[DEBUG] Error in result for seed {seed_id}: {result['error']}")
            logger.error(
                f"Failed to update segment statistics for seed {seed_id}: {result['error']}"
            )
            return False

        if "skeleton_path_length_mm" in result:
            print(
                f"[DEBUG] Successfully updated skeleton length for seed {seed_id}: "  # pylint: disable=line-too-long
                f"{result['skeleton_path_length_mm']:.2f} mm"  # pylint: disable=line-too-long
            )  # pylint: disable=line-too-long
            logger.info(
                f"Updated skeleton length for seed {seed_id}: "
                f"{result['skeleton_path_length_mm']:.2f} mm"
            )
        else:
            print(f"[DEBUG] No skeleton length returned for seed {seed_id}")
            logger.info(f"Completed update for seed {seed_id} (no skeleton length returned)")

        print(f"[DEBUG] Successfully processed seed {seed_id}")
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[DEBUG] Exception processing skeleton update for seed {seed_id}: {e}")
        logger.error(f"Error processing skeleton update for seed {seed_id}: {e}")
        return False


def run_skeleton_update_worker(  # pylint: disable=too-many-branches, too-many-statements
    project_name: str,
    user_id: str = "skeleton_update_worker",
    polling_period: float = 10.0,
    server_address: str = "https://proofreading.zetta.ai",
    max_retries: int = 5,
    cleanup_interval_hours: int = 24,
    completed_cleanup_days: int = 7,
) -> None:
    """
    Run the skeleton update background worker.
    
    Args:
        project_name: Project name to process updates for
        user_id: User ID for the worker
        polling_period: How often to poll for new updates (seconds)
        server_address: Cave server address
        max_retries: Maximum retries before marking as permanently failed
        cleanup_interval_hours: How often to run cleanup (hours)
        completed_cleanup_days: Remove completed updates older than this many days
    """
    print("[DEBUG] Starting skeleton update worker")
    print(
        f"[DEBUG] Worker config: project={project_name}, user={user_id}, "  # pylint: disable=line-too-long
        f"polling={polling_period}s"  # pylint: disable=line-too-long
    )  # pylint: disable=line-too-long
    print(f"[DEBUG] Server: {server_address}, max_retries={max_retries}")  # pylint: disable=line-too-long
    print(
        f"[DEBUG] Cleanup: interval={cleanup_interval_hours}h, "  # pylint: disable=line-too-long
        f"cleanup_days={completed_cleanup_days}"  # pylint: disable=line-too-long
    )  # pylint: disable=line-too-long

    logger.info(
        f"Starting skeleton update worker for project '{project_name}' "
        f"with user '{user_id}' (polling every {polling_period}s)"
    )

    processed_count = 0
    last_cleanup = time.time()
    cleanup_interval_sec = cleanup_interval_hours * 3600
    loop_count = 0

    print("[DEBUG] Starting main worker loop")
    try: # pylint: disable=too-many-nested-blocks
        while True:
            loop_count += 1
            try:
                print(f"[DEBUG] Worker loop #{loop_count} - checking for pending updates")
                # Get next pending update
                update_entry = get_next_pending_update(project_name=project_name)
                print(f"[DEBUG] get_next_pending_update returned: {update_entry}")

                if update_entry:
                    seed_id = update_entry["seed_id"]
                    processed_count += 1

                    print(
                        f"[DEBUG] Found pending update for seed {seed_id}, "  # pylint: disable=line-too-long
                        f"retry count: {update_entry['retry_count']}"  # pylint: disable=line-too-long
                    )  # pylint: disable=line-too-long
                    logger.info(
                        f"Processing update #{processed_count}: seed {seed_id} "
                        f"(retry {update_entry['retry_count']})"
                    )

                    # Mark as processing
                    print(f"[DEBUG] Marking seed {seed_id} as processing")
                    if not mark_update_processing(project_name=project_name, seed_id=seed_id):
                        print(f"[DEBUG] Failed to mark seed {seed_id} as processing")
                        logger.warning(f"Could not mark seed {seed_id} as processing")
                        continue

                    print(
                        f"[DEBUG] Successfully marked seed {seed_id} as processing, "  # pylint: disable=line-too-long
                        "starting update"  # pylint: disable=line-too-long
                    )  # pylint: disable=line-too-long
                    # Process the update
                    success = process_skeleton_update(
                        project_name=project_name,
                        seed_id=seed_id,
                        server_address=server_address
                    )

                    if success:
                        print(
                            f"[DEBUG] Processing succeeded for seed {seed_id}, marking as completed"  # pylint: disable=line-too-long
                        )  # pylint: disable=line-too-long
                        # Mark as completed
                        mark_update_completed(project_name=project_name, seed_id=seed_id)
                        logger.info(f"Successfully completed skeleton update for seed {seed_id}")
                    else:
                        print(f"[DEBUG] Processing failed for seed {seed_id}, marking as failed")
                        # Mark as failed (will retry or permanently fail based on retry count)
                        error_msg = f"Cave API call failed for seed {seed_id}"
                        mark_update_failed(
                            project_name=project_name,
                            seed_id=seed_id,
                            error_message=error_msg,
                            max_retries=max_retries
                        )

                        # Add exponential backoff delay for retries
                        retry_count = update_entry["retry_count"]
                        if retry_count < max_retries:
                            # Exponential backoff: 2^retry_count seconds (capped at 300s)
                            backoff_delay = min(2 ** retry_count, 300)
                            print(f"[DEBUG] Waiting {backoff_delay}s for exponential backoff")
                            logger.info(f"Waiting {backoff_delay}s before next attempt")
                            time.sleep(backoff_delay)

                else:
                    print(f"[DEBUG] No pending updates found, sleeping for {polling_period}s")
                    # No pending updates, wait before checking again
                    time.sleep(polling_period)

                # Periodic cleanup of completed entries
                current_time = time.time()
                if current_time - last_cleanup > cleanup_interval_sec:
                    try:
                        cleaned_count = cleanup_completed_updates(
                            project_name=project_name,
                            days_old=completed_cleanup_days
                        )
                        if cleaned_count > 0:
                            logger.info(f"Cleaned up {cleaned_count} old completed entries")
                        last_cleanup = current_time
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.error(f"Error during cleanup: {e}")

                # Log queue stats periodically
                if processed_count % 10 == 0 and processed_count > 0:
                    try:
                        stats = get_queue_stats(project_name=project_name)
                        logger.info(f"Queue stats: {stats}")
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.error(f"Error getting queue stats: {e}")

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retrying

    except KeyboardInterrupt:
        logger.info(f"Worker stopped by user after processing {processed_count} updates")


@click.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project to process skeleton updates for"
)
@click.option(
    "--user_id", "-u",
    default="skeleton_update_worker",
    help="User ID for the worker (default: skeleton_update_worker)"
)
@click.option(
    "--polling_period", "-t",
    type=float,
    default=10.0,
    help="Polling period in seconds (default: 10.0)"
)
@click.option(
    "--server_address", "-s",
    default="https://proofreading.zetta.ai",
    help="Cave server address (default: https://proofreading.zetta.ai)"
)
@click.option(
    "--max_retries", "-r",
    type=int,
    default=5,
    help="Maximum retries before permanent failure (default: 5)"
)
@click.option(
    "--cleanup_interval_hours",
    type=int,
    default=24,
    help="Hours between cleanup runs (default: 24)"
)
@click.option(
    "--completed_cleanup_days",
    type=int,
    default=7,
    help="Remove completed entries older than this many days (default: 7)"
)
def main(
    project_name: str,
    user_id: str,
    polling_period: float,
    server_address: str,
    max_retries: int,
    cleanup_interval_hours: int,
    completed_cleanup_days: int,
) -> None:
    """Run the skeleton update background worker."""
    run_skeleton_update_worker(
        project_name=project_name,
        user_id=user_id,
        polling_period=polling_period,
        server_address=server_address,
        max_retries=max_retries,
        cleanup_interval_hours=cleanup_interval_hours,
        completed_cleanup_days=completed_cleanup_days,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
