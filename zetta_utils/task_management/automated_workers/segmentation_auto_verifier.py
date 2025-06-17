"""
Segmentation Auto Verifier Worker

An automated worker that polls for tasks, processes them by parsing the ng_state,
and automatically releases them with "pass" status.
"""

import os
import time

import attrs
import click
import meshparty
import pcg_skel
from caveclient import CAVEclient
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception

from zetta_utils import log
from zetta_utils.layer.volumetric.cloudvol.build import build_cv_layer
from zetta_utils.task_management.task import get_task, release_task, start_task
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.db.models import TimesheetModel
from sqlalchemy import select

slack_client = WebClient(token=os.environ["ZETTA_PROOFREDING_BOT_SLACK_TOKEN"])

logger = log.get_logger()
console = Console()


@attrs.mutable
class SegmentSkeleton:
    """Class to hold segment ID and skeleton data."""
    segment_id: int = attrs.field()
    skeleton: meshparty.skeleton.Skeleton = attrs.field()



def create_task_dashboard_link(project_name: str, task_id: str) -> str:
    """Create a task dashboard link."""
    dashboard_url = f"https://zetta-task-manager.vercel.app/{project_name}/tasks/{task_id}"
    logger.info(f"Created task dashboard link: {dashboard_url}")
    console.print(f"[dim]Task dashboard link: {dashboard_url}[/dim]")
    return dashboard_url


def is_unavailable_error(exception):
    """Check if the exception contains 'unavailable' (case insensitive)."""
    error_message = str(exception).lower()
    return "unavailable" in error_message


def log_retry_attempt(retry_state):
    """Log retry attempts for debugging."""
    if retry_state.outcome is not None and retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number
        logger.warning(
            f"Skeleton fetch attempt {attempt_number} failed with: {exception}. Retrying..."
        )
        console.print(
            f"[yellow]⚠️  Skeleton unavailable (attempt {attempt_number}), retrying...[/yellow]"
        )


def extract_target_location(ng_dict: dict) -> list | None:
    """Extract the target location from the neuroglancer state."""
    for layer in ng_dict["layers"]:
        if layer.get("name") == "Target Location" and "annotations" in layer:
            annotations = layer["annotations"]
            if annotations and len(annotations) > 0:
                point = annotations[0].get("point")
                if point:
                    point[-1] -= 0.5
                    return point
    return None


@retry(
    retry=retry_if_exception(is_unavailable_error),
    stop=stop_after_delay(300),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    reraise=True
)
def get_skeleton(
    ng_dict: dict, graphene_path: str, ng_resolution: list, sv_resolution: list
) -> SegmentSkeleton | None:
    """Get skeleton from segmentation layer using target location.

    Automatically retries if the error contains 'unavailable' (case insensitive)
    for up to 5 minutes with exponential backoff.
    """
    # Extract target location from neuroglancer state
    target_location = extract_target_location(ng_dict)
    if not target_location:
        logger.warning("No target location found in neuroglancer state")
        return None

    logger.info(f"Found target location: {target_location}")
    console.print(f"[dim]Target location: {target_location}[/dim]")

    # Create CloudVolume for supervoxel layer
    sv_layer = build_cv_layer(
        path=graphene_path,
        cv_kwargs={"agglomerate": True},
        index_resolution=ng_resolution,
        default_desired_resolution=sv_resolution,
        allow_slice_rounding=True
    )
    # Convert float coordinates to integer voxel coordinates
    voxel_coords = [int(coord) for coord in target_location[:3]]

    # Read supervoxel at target location
    logger.info(f"Reading supervoxel at voxel coordinates: {voxel_coords}")
    console.print(f"[dim]Reading supervoxel at: {voxel_coords}[/dim]")

    sv_data = sv_layer[voxel_coords[0]:voxel_coords[0]+1,
                  voxel_coords[1]:voxel_coords[1]+1,
                  voxel_coords[2]:voxel_coords[2]+1]

    segment_id = int(sv_data[0, 0, 0])
    logger.info(f"Segment {segment_id}")
    console.print(f"[dim]Segment {segment_id}[/dim]")

    # Get skeleton using CAVE client
    client = CAVEclient(
        datastack_name="kronauer_ant",
        server_address="https://proofreading.zetta.ai",
        auth_token_file="~/.cloudvolume/secrets/cave-secret.json"
    )

    try:
        skel = pcg_skel.pcg_skeleton(root_id=segment_id, client=client)
        return SegmentSkeleton(segment_id=segment_id, skeleton=skel)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Failed to get skeleton for segment {segment_id}: {e}")
        raise


def process_task(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    task_id: str, project_name: str, user_id: str,
    min_skeleton_length_mm: int, slack_channel: str, task_count: int,
    graphene_path: str, ng_resolution: list, sv_resolution: list, slack_users: tuple
) -> None:
    """Process a single task - verify skeleton and send notifications."""
    # Create processing panel
    processing_text = Text(f"Processing Task #{task_count}", style="bold green")
    processing_panel = Panel(
        f"[cyan]Task ID:[/cyan] {task_id}\n"
        f"[cyan]Status:[/cyan] Analyzing ng_state...",
        title=processing_text,
        border_style="green"
    )
    console.print(processing_panel)

    # Get task details
    task_details = get_task(project_name=project_name, task_id=task_id)
    task_type = task_details.get("task_type", "unknown")

    logger.info(f"Processing task {task_id} (type: {task_type})")
    console.print(f"[cyan]Task Type:[/cyan] {task_type}")

    # Get trace task information
    ng_state = task_details["ng_state"]
    extra_data = task_details.get('extra_data')
    if not extra_data or 'trace_task_id' not in extra_data:
        logger.error(f"Task {task_id} missing trace_task_id in extra_data")
        return
    trace_task_id = extra_data['trace_task_id']
    trace_task_details = get_task(project_name=project_name, task_id=trace_task_id)
    logger.info(f"Successfully parsed ng_state for task {task_id}")

    # Process skeleton verification
    try:
        skeleton_info = get_skeleton(ng_state, graphene_path, ng_resolution, sv_resolution)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(e)
        logger.error(f"Error getting skeleton for task {task_id}: {e}")
        skeleton_info = None

    skeleton_length_mm = None
    verification_pass = False

    if skeleton_info:
        skeleton_length_nm = skeleton_info.skeleton.path_length()
        skeleton_length_mm = skeleton_length_nm / 1_000_000
        verification_pass = skeleton_length_mm >= min_skeleton_length_mm

    # Get timesheet summary
    timesheet_summary = get_task_timesheet_summary(
        project_name=project_name, task_id=trace_task_id
    )

    # Build Slack message
    trace_dashboard_link = create_task_dashboard_link(project_name, trace_task_id)
    verify_dashboard_link = create_task_dashboard_link(project_name, task_id)

    # Header with task info and verification status
    header_icon = "✅" if verification_pass else "❌"

    summary_message = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    summary_message += f"{header_icon} *Task Completed*\n"
    summary_message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    summary_message += f"*Trace Task:* `{trace_task_id}` [<{trace_dashboard_link}|link>]\n"
    summary_message += f"*Verification Task:* `{task_id}` [<{verify_dashboard_link}|link>]\n"
    summary_message += f"*Completed by:* {trace_task_details['completed_user_id']}\n\n"

    # Skeleton verification section
    if skeleton_length_mm is not None:
        if verification_pass:
            summary_message += (
                f"*Skeleton Length:* {skeleton_length_mm:.2f} mm "
                f"(minimum: {min_skeleton_length_mm} mm) ✅\n"
            )
        else:
            summary_message += (
                f"*Skeleton Length:* {skeleton_length_mm:.2f} mm "
                f"(minimum: {min_skeleton_length_mm} mm) ❌\n"
            )
            summary_message += "⚠️ *BELOW MINIMUM REQUIREMENT*\n"
    else:
        summary_message += "*No skeleton found in segmentation* ⚠️\n"

    # Time tracking section
    if timesheet_summary['total_seconds'] > 0:
        summary_message += f"\n*Time Spent:* {timesheet_summary['formatted_total']}"

    # Send Slack notification
    logger.info(f"Sending Slack message for completed task {trace_task_id}")
    if slack_channel:
        try:
            # Send main message
            response = slack_client.chat_postMessage(
                channel=slack_channel,
                text=summary_message,
                unfurl_links=False,
                unfurl_media=False
            )
            console.print(f"[green]✅ Sent Slack notification to {slack_channel}[/green]")

            # If verification failed and we have users to tag, create a thread
            if not verification_pass and slack_users:
                thread_ts = response['ts']
                # Create user mentions
                user_mentions = ' '.join([f"<@{user}>" for user in slack_users])
                thread_message = (
                    f"{user_mentions} This task failed skeleton verification. Please review."
                )

                slack_client.chat_postMessage(
                    channel=slack_channel,
                    thread_ts=thread_ts,
                    text=thread_message
                )
                console.print(
                    f"[yellow]📢 Tagged users in thread: {', '.join(slack_users)}[/yellow]"
                )

        except SlackApiError as e:
            logger.error(f"Failed to send Slack message: {e}")
            console.print(f"[red]❌ Failed to send Slack notification: {e}[/red]")

    # Display skeleton info in console
    if skeleton_info:
        console.print(f"[green]✅ Got skeleton for segment {skeleton_info.segment_id}[/green]")
        console.print(f"[cyan]Skeleton length:[/cyan] {skeleton_length_mm:.2f}mm")
        console.print(f"[cyan]Minimum required:[/cyan] {min_skeleton_length_mm}mm")
        console.print(f"[cyan]Verification:[/cyan] {'PASS' if verification_pass else 'FAIL'}")
    else:
        console.print("[yellow]⚠️  No skeleton found in neuroglancer state[/yellow]")

    console.print(f"\n[bold cyan]Completed verification task:[/bold cyan] {task_id}")

    # Release task with appropriate status
    completion_status = "pass" if verification_pass else "fail"
    logger.info(f"Releasing task `{task_id}` with `{completion_status}` status")
    release_task(
        project_name=project_name,
        task_id=task_id,
        user_id=user_id,
        completion_status=completion_status
    )


def get_task_timesheet_summary(project_name: str, task_id: str) -> dict:
    """Get timesheet summary for a specific task.

    Returns dict with:
    - total_seconds: Total time spent on the task
    - user_breakdown: Dict of user_id to seconds spent
    - formatted_total: Human-readable total time
    - formatted_breakdown: List of formatted strings for each user
    """
    with get_session_context() as session:
        # Query all timesheet entries for this task
        query = (
            select(TimesheetModel)
            .where(TimesheetModel.project_name == project_name)
            .where(TimesheetModel.task_id == task_id)
        )

        timesheets = session.execute(query).scalars().all()

        # Calculate totals
        total_seconds = 0
        user_breakdown: dict[str, int] = {}

        for timesheet in timesheets:
            total_seconds += timesheet.seconds_spent
            user_breakdown[timesheet.user] = (
                user_breakdown.get(timesheet.user, 0) + timesheet.seconds_spent
            )

        # Format time strings
        def format_duration(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60

            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"

        formatted_breakdown = []
        for user_id, seconds in sorted(user_breakdown.items()):
            formatted_breakdown.append(f"• {user_id}: {format_duration(seconds)}")

        return {
            "total_seconds": total_seconds,
            "user_breakdown": user_breakdown,
            "formatted_total": format_duration(total_seconds),
            "formatted_breakdown": formatted_breakdown
        }


@click.command()
@click.option(
    "--user_id", "-u",
    required=True,
    help="User ID for the automated worker"
)
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project to process tasks from"
)
@click.option(
    "--polling_period", "-t",
    type=float,
    default=5.0,
    help="Polling period in seconds (default: 5.0)"
)
@click.option(
    "--min_skeleton_length_mm", "-m",
    type=int,
    default=0,
    help="Minimum skeleton length in millimeters (default: 0)"
)
@click.option(
    "--slack_channel", "-c",
    help="Slack channel to post messages to"
)
@click.option(
    "--graphene_path", "-g",
    required=True,
    help="Graphene segmentation path"
)
@click.option(
    "--ng_resolution", "-nr",
    nargs=3,
    type=int,
    required=True,
    help="Neuroglancer state resolution (e.g. 8 8 42)"
)
@click.option(
    "--sv_resolution", "-sr",
    nargs=3,
    type=int,
    required=True,
    help="Supervoxel resolution (e.g. 16 16 42)"
)
@click.option(
    "--slack_users", "-su",
    multiple=True,
    help="Slack user IDs to tag on failure (can be specified multiple times)"
)
def run_worker(
    user_id: str, project_name: str, polling_period: float, min_skeleton_length_mm: int,
    slack_channel: str, graphene_path: str, ng_resolution: tuple, sv_resolution: tuple,
    slack_users: tuple
):
    """Run the segmentation auto verifier worker."""
    # Display startup banner
    users_info = (
        f"[yellow]Slack Users to Tag:[/yellow] "
        f"{', '.join(slack_users) if slack_users else 'None'}\n"
    ) if slack_channel else ""
    startup_panel = Panel(
        f"[bold cyan]Segmentation Auto Verifier Worker[/bold cyan]\n\n"
        f"[yellow]User ID:[/yellow] {user_id}\n"
        f"[yellow]Project:[/yellow] {project_name}\n"
        f"[yellow]Polling Period:[/yellow] {polling_period}s\n"
        f"[yellow]Min Skeleton Length:[/yellow] {min_skeleton_length_mm}mm\n"
        f"[yellow]Graphene Path:[/yellow] {graphene_path}\n"
        f"[yellow]NG Resolution:[/yellow] {ng_resolution}\n"
        f"[yellow]SV Resolution:[/yellow] {sv_resolution}\n"
        f"{users_info}\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        title="🤖 Worker Starting",
        border_style="blue"
    )
    console.print(startup_panel)
    logger.info(
        f"Starting segmentation auto verifier worker for project '{project_name}' "
        f"with user '{user_id}'"
    )

    had_task_last_time = False
    task_count = 0

    try:
        while True:
            task_id = None
            try:
                # Try to start a task
                logger.debug(f"Polling for tasks (user: {user_id}, project: {project_name})")
                task_id = start_task(
                    project_name=project_name,
                    user_id=user_id
                )
                if task_id:
                    task_count += 1
                    logger.info(f"Started processing task {task_id}")

                    process_task(
                        task_id, project_name, user_id,
                        min_skeleton_length_mm, slack_channel, task_count,
                        graphene_path, list(ng_resolution), list(sv_resolution), slack_users
                    )

                    had_task_last_time = True

                else:
                    # Only log when transitioning from having tasks to no tasks
                    if had_task_last_time:
                        logger.info("No tasks available, entering wait mode")
                    console.print("[dim]⏳ No tasks available, waiting...[/dim]")
                    had_task_last_time = False

            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
                console.print(f"[red]❌ Error processing task: {e}[/red]")
                had_task_last_time = False

            # Sleep only if we didn't have a task last time
            if not had_task_last_time:
                time.sleep(polling_period)

    except KeyboardInterrupt:
        logger.info(f"Worker stopped by user after processing {task_count} tasks")
        shutdown_panel = Panel(
            f"[bold red]Worker Shutdown[/bold red]\n\n"
            f"[yellow]Tasks Processed:[/yellow] {task_count}\n"
            f"[dim]Thank you for using the auto verifier![/dim]",
            title="🛑 Goodbye",
            border_style="red"
        )
        console.print(shutdown_panel)


if __name__ == "__main__":
    run_worker()  # pylint: disable=no-value-for-parameter
