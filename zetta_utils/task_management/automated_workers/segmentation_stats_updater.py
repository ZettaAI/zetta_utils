"""
Segmentation Stats Updater Worker

An automated worker that polls for seg_stats_update_v0 tasks and updates segment statistics
(skeleton length and synapse counts) from CAVE.
"""

import os
import time
import traceback
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from zetta_utils import log
from zetta_utils.task_management.segment import update_segment_statistics
from zetta_utils.task_management.task import get_task, release_task, start_task

logger = log.get_logger()
console = Console()


def get_slack_client() -> WebClient | None:
    """Get Slack client if token is available."""
    token = os.environ.get("ZETTA_PROOFREDING_BOT_SLACK_TOKEN")
    if token:
        return WebClient(token=token)
    return None


def create_task_dashboard_link(project_name: str, task_id: str) -> str:
    """Create a task dashboard link."""
    dashboard_url = f"https://zetta-task-manager.vercel.app/{project_name}/tasks/{task_id}"
    logger.info(f"Created task dashboard link: {dashboard_url}")
    console.print(f"[dim]Task dashboard link: {dashboard_url}[/dim]")
    return dashboard_url


def send_slack_error_notification(
    slack_channel: str | None, project_name: str, error_message: str, duration_minutes: int
) -> None:
    """Send error notification to Slack channel."""
    if not slack_channel:
        return

    slack_client = get_slack_client()
    if not slack_client:
        return

    try:
        message = (
            f"🚨 *Segmentation Stats Updater Worker Error*\n\n"
            f"*Project:* {project_name}\n"
            f"*Duration:* {duration_minutes} minutes of continuous errors\n"
            f"*Error:* ```{error_message}```\n"
            f"The worker is still running but encountering repeated errors."
        )

        slack_client.chat_postMessage(
            channel=slack_channel, text=message, unfurl_links=False, unfurl_media=False
        )
        console.print(
            f"[yellow]📢 Sent error notification to Slack channel {slack_channel}[/yellow]"
        )

    except SlackApiError as e:
        logger.error(f"Failed to send Slack error notification: {e}")
        console.print(f"[red]❌ Failed to send Slack notification: {e}[/red]")


def process_task(task_id: str, project_name: str, user_id: str, task_count: int) -> None:
    """Process a single seg_stats_update_v0 task."""
    # Create processing panel
    processing_text = Text(f"Processing Task #{task_count}", style="bold green")
    processing_panel = Panel(
        f"[cyan]Task ID:[/cyan] {task_id}\n"
        f"[cyan]Status:[/cyan] Updating segment statistics...",
        title=processing_text,
        border_style="green",
    )
    console.print(processing_panel)

    # Get task details
    task_details = get_task(project_name=project_name, task_id=task_id)
    task_type = task_details.get("task_type", "unknown")

    logger.info(f"Processing task {task_id} (type: {task_type})")
    console.print(f"[cyan]Task Type:[/cyan] {task_type}")

    # Extract seed_id from task extra_data
    extra_data = task_details.get("extra_data")
    if not extra_data:
        logger.error(f"Task {task_id} missing extra_data")
        console.print("[red]❌ Task missing extra_data, cannot process[/red]")
        release_task(
            project_name=project_name,
            task_id=task_id,
            user_id=user_id,
            completion_status="Done",
        )
        return
    seed_id = extra_data.get("seed_id")

    if not seed_id:
        logger.error(f"Task {task_id} missing seed_id in extra_data")
        console.print("[red]❌ Task missing seed_id, cannot process[/red]")
        release_task(
            project_name=project_name,
            task_id=task_id,
            user_id=user_id,
            completion_status="Done",  # Mark as done even though we couldn't process
        )
        return

    console.print(f"[cyan]Seed ID:[/cyan] {seed_id}")

    # Update segment statistics
    try:
        logger.info(f"Updating statistics for segment with seed_id {seed_id}")
        results = update_segment_statistics(project_name=project_name, seed_id=seed_id)

        # Display results
        console.print("\n[green]✅ Statistics Updated:[/green]")

        if "skeleton_path_length_mm" in results:
            console.print(
                f"  [cyan]Skeleton length:[/cyan] {results['skeleton_path_length_mm']:.2f} mm"
            )
        elif "skeleton_error" in results:
            console.print(f"  [yellow]Skeleton error:[/yellow] {results['skeleton_error']}")

        if "pre_synapse_count" in results:
            console.print(f"  [cyan]Pre-synaptic count:[/cyan] {results['pre_synapse_count']}")
            console.print(f"  [cyan]Post-synaptic count:[/cyan] {results['post_synapse_count']}")
        elif "synapse_error" in results:
            console.print(f"  [yellow]Synapse error:[/yellow] {results['synapse_error']}")

        if "error" in results:
            console.print(f"  [red]Error:[/red] {results['error']}")

        logger.info(f"Successfully updated statistics for segment {seed_id}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to update statistics for segment {seed_id}: {e}")
        console.print(f"[red]❌ Failed to update statistics: {e}[/red]")
        # Still release the task as done to avoid getting stuck

    # Release task as done
    logger.info(f"Releasing task {task_id} with 'Done' status")
    release_task(
        project_name=project_name, task_id=task_id, user_id=user_id, completion_status="Done"
    )

    console.print(f"\n[bold cyan]Completed stats update task:[/bold cyan] {task_id}")


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
@click.option("--slack_channel", "-c", help="Slack channel to post error messages to")
def run_worker(  # pylint: disable=too-many-statements
    user_id: str, project_name: str, polling_period: float, slack_channel: str
):
    """Run the segmentation stats updater worker."""
    # Display startup banner
    startup_panel = Panel(
        f"[bold cyan]Segmentation Stats Updater Worker[/bold cyan]\n\n"
        f"[yellow]User ID:[/yellow] {user_id}\n"
        f"[yellow]Project:[/yellow] {project_name}\n"
        f"[yellow]Polling Period:[/yellow] {polling_period}s\n"
        f"[yellow]Slack Channel:[/yellow] {slack_channel or 'None'}\n\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        title="🤖 Worker Starting",
        border_style="blue",
    )
    console.print(startup_panel)
    logger.info(
        f"Starting segmentation stats updater worker for project '{project_name}' "
        f"with user '{user_id}'"
    )

    had_task_last_time = False
    task_count = 0

    # Exception tracking
    exception_start_time = None
    last_exception = None
    slack_notification_sent = False

    # Retry constants
    RETRY_DELAY_SECONDS = 60  # 1 minute
    SLACK_NOTIFICATION_THRESHOLD_MINUTES = 30  # 30 minutes

    try:
        while True:
            task_id = None
            try:
                # Try to start a task
                logger.debug(f"Polling for tasks (user: {user_id}, project: {project_name})")
                task_id = start_task(project_name=project_name, user_id=user_id)

                if task_id:
                    task_count += 1
                    logger.info(f"Started processing task {task_id}")

                    process_task(task_id, project_name, user_id, task_count)

                    had_task_last_time = True

                    # Reset exception tracking on successful task
                    exception_start_time = None
                    last_exception = None
                    slack_notification_sent = False

                else:
                    if (
                        had_task_last_time
                    ):  # Only log when transitioning from having tasks to no tasks
                        logger.info("No tasks available, entering wait mode")
                    console.print("[dim]⏳ No tasks available, waiting...[/dim]")
                    had_task_last_time = False

                    # Reset exception tracking when no tasks
                    exception_start_time = None
                    last_exception = None
                    slack_notification_sent = False

            except Exception as e:  # pylint: disable=broad-exception-caught
                # Track exception timing
                current_time = datetime.now()

                if exception_start_time is None:
                    exception_start_time = current_time

                # Calculate duration of exceptions
                exception_duration = current_time - exception_start_time
                duration_minutes = int(exception_duration.total_seconds() / 60)

                # Log the error
                error_traceback = traceback.format_exc()
                logger.error(f"Error processing task {task_id}: {e}\n{error_traceback}")
                console.print(f"[red]❌ Error: {e}[/red]")
                console.print(
                    f"[yellow]⏱️  Exception duration: {duration_minutes} minutes[/yellow]"
                )

                # Store last exception
                last_exception = str(e)

                # Send Slack notification after threshold
                if (
                    duration_minutes >= SLACK_NOTIFICATION_THRESHOLD_MINUTES
                    and not slack_notification_sent
                    and slack_channel
                ):
                    send_slack_error_notification(
                        slack_channel, project_name, last_exception, duration_minutes
                    )
                    slack_notification_sent = True

                # Pause for retry delay
                console.print(
                    f"[yellow]😴 Pausing for {RETRY_DELAY_SECONDS} seconds before retry...[/yellow]"
                )
                time.sleep(RETRY_DELAY_SECONDS)

                had_task_last_time = False
                continue

            # Sleep only if we didn't have a task and no exception
            if not had_task_last_time:
                time.sleep(polling_period)

    except KeyboardInterrupt:
        logger.info(f"Worker stopped by user after processing {task_count} tasks")
        shutdown_panel = Panel(
            f"[bold red]Worker Shutdown[/bold red]\n\n"
            f"[yellow]Tasks Processed:[/yellow] {task_count}\n"
            f"[dim]Thank you for using the stats updater![/dim]",
            title="🛑 Goodbye",
            border_style="red",
        )
        console.print(shutdown_panel)


if __name__ == "__main__":
    run_worker()  # pylint: disable=no-value-for-parameter
