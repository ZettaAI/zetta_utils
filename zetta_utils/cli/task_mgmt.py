"""
Task management CLI commands for zetta_utils.

Usage:
    zetta task_mgmt start -p <project> -u <user> [-t <task>]
    zetta task_mgmt get -p <project> -t <task>
    zetta task_mgmt release -p <project> -u <user> -c <status> [-t <task>]
    zetta task_mgmt tasks -p <project>
    zetta task_mgmt jobs -p <project>
    zetta task_mgmt reactivate -p <project> -t <task>
    zetta task_mgmt clear -p <project> [-u] [-t] [-f]

Shorthand options:
    -p, --project_name    Name of the project
    -u, --user_id         User ID (or --include-users for clear)
    -t, --task_id         Task ID (or --include-task-types for clear)
    -c, --completion_status   Completion status
    -f, --force           Skip confirmation prompt (clear only)
"""

import json
import time
import urllib.parse
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from zetta_utils.task_management.job import list_jobs_summary
from zetta_utils.task_management.task import (
    get_task,
    list_tasks_summary,
    reactivate_task,
    release_task,
    start_task,
)
from zetta_utils.task_management.user import get_user
from zetta_utils.task_management.clear_project import (
    clear_project_data,
    clear_project_users,
    clear_project_task_types,
)
from zetta_utils.task_management.segment_link import get_segment_link

console = Console()


def print_timing(func_name: str, elapsed_time: float):
    """Print the timing information for a command."""
    console.print(f"\n⏱️  {func_name} completed in {elapsed_time:.3f} seconds", style="dim")


def create_neuroglancer_link(ng_state: str) -> str:
    """Create a neuroglancer link from ng_state."""
    encoded_state = urllib.parse.quote(json.dumps(ng_state), safe='')
    return f"https://spelunker.cave-explorer.org/#!{encoded_state}"


def print_task_details(task_details: dict):
    """Print task details in a formatted table using rich."""
    table = Table(title="Task Details", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True, width=20)
    table.add_column("Value", style="white", no_wrap=False, overflow="fold")

    # Define field order and display names
    field_mapping = {
        "task_id": "Task ID",
        "job_id": "Job ID",
        "task_type": "Task Type",
        "completion_status": "Completion Status",
        "assigned_user_id": "Assigned User",
        "active_user_id": "Active User",
        "completed_user_id": "Completed User",
        "priority": "Priority",
        "batch_id": "Batch ID",
        "last_leased_ts": "Last Leased",
        "is_active": "Is Active",
        "is_paused": "Is Paused",
    }

    # Add basic fields
    for field, display_name in field_mapping.items():
        if field in task_details:
            value = str(task_details[field])
            # Special formatting for some fields
            if field == "is_active":
                value = "✅ Yes" if task_details[field] else "❌ No"
            elif field == "is_paused":
                value = "⏸️ Yes" if task_details[field] else "▶️ No"
            elif field == "completion_status" and not task_details[field]:
                value = "⏳ In Progress"

            table.add_row(display_name, value)

    console.print(table)

    # Add neuroglancer link outside table for easy clicking
    if "ng_state" in task_details and task_details["ng_state"]:
        ng_link = create_neuroglancer_link(task_details["ng_state"])
        console.print(f"\n🔗 Neuroglancer Link:", style="bold cyan")
        console.print(ng_link, style="blue underline")


@click.group()
def task_mgmt():
    """Task management commands."""
    pass


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project to start a task from"
)
@click.option(
    "--user_id", "-u",
    required=True,
    help="ID of the user starting the task"
)
@click.option(
    "--task_id", "-t",
    help="Optional specific task ID to start (if not provided, auto-selects a task)",
    default=None
)
def start(project_name: str, user_id: str, task_id: Optional[str]):
    """Start a task from a project."""
    start_time = time.time()

    try:
        result_task_id = start_task(
            project_name=project_name,
            user_id=user_id,
            task_id=task_id
        )

        elapsed_time = time.time() - start_time

        if result_task_id is None:
            console.print("No task available to start for this user.", style="yellow")
            print_timing("start_task", elapsed_time)
            return

        console.print(f"✅ Successfully started task: {result_task_id}", style="green")

        # Get and display task details
        task_details = get_task(
            project_name=project_name,
            task_id=result_task_id
        )
        print_task_details(task_details)
        print_timing("start_task", elapsed_time)

    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error starting task: {e}", style="red")
        print_timing("start_task", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project"
)
@click.option(
    "--task_id", "-t",
    required=True,
    help="ID of the task to get"
)
def get(project_name: str, task_id: str):
    """Get task details by ID."""
    start_time = time.time()

    try:
        task_details = get_task(
            project_name=project_name,
            task_id=task_id
        )
        elapsed_time = time.time() - start_time
        console.print(f"✅ Retrieved task: {task_id}", style="green")
        print_task_details(task_details)
        print_timing("get_task", elapsed_time)

    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error getting task: {e}", style="red")
        print_timing("get_task", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project"
)
@click.option(
    "--task_id", "-t",
    required=False,
    help="ID of the task to release (if not provided, releases user's active task)"
)
@click.option(
    "--user_id", "-u",
    required=True,
    help="ID of the user releasing the task"
)
@click.option(
    "--completion_status", "-c",
    required=True,
    help="Completion status to set for the task"
)
def release(project_name: str, task_id: Optional[str], user_id: str, completion_status: str):
    """Release a task with completion status."""
    start_time = time.time()

    try:
        # If no task_id provided, get user's active task
        if task_id is None:
            user = get_user(project_name=project_name, user_id=user_id)
            if not user["active_task"]:
                elapsed_time = time.time() - start_time
                console.print(f"❌ User {user_id} does not have an active task", style="red")
                print_timing("release_task", elapsed_time)
                return
            
            task_id = user["active_task"]
            console.print(f"📋 Releasing user's active task: {task_id}", style="cyan")

        success = release_task(
            project_name=project_name,
            task_id=task_id,
            user_id=user_id,
            completion_status=completion_status
        )

        elapsed_time = time.time() - start_time

        if success:
            console.print(f"✅ Successfully released task: {task_id} with status: {completion_status}", style="green")

            # Get and display updated task details
            task_details = get_task(
                project_name=project_name,
                task_id=task_id
            )
            print_task_details(task_details)
        else:
            console.print(f"❌ Failed to release task: {task_id}", style="red")

        print_timing("release_task", elapsed_time)

    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error releasing task: {e}", style="red")
        print_timing("release_task", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project"
)
def tasks(project_name: str):
    """List task counts and sample task IDs for a project."""
    start_time = time.time()

    try:
        summary = list_tasks_summary(project_name=project_name)

        elapsed_time = time.time() - start_time

        # Create summary table
        summary_table = Table(title=f"Task Summary for {project_name}", show_header=True, header_style="bold magenta")
        summary_table.add_column("Category", style="cyan", no_wrap=True)
        summary_table.add_column("Count", style="yellow", justify="right")

        summary_table.add_row("Active (incomplete)", str(summary["active_count"]))
        summary_table.add_row("Active (completed)", str(summary["completed_count"]))
        summary_table.add_row("Paused", str(summary["paused_count"]))

        console.print(summary_table)

        # Create active unpaused tasks table
        if summary["active_unpaused_ids"]:
            unpaused_table = Table(title="First 5 Active Unpaused Tasks", show_header=True, header_style="bold green")
            unpaused_table.add_column("Task ID", style="white")

            for task_id in summary["active_unpaused_ids"]:
                unpaused_table.add_row(task_id)

            console.print(unpaused_table)
        else:
            console.print("📝 No active unpaused tasks found", style="dim")

        # Create active paused tasks table
        if summary["active_paused_ids"]:
            paused_table = Table(title="First 5 Active Paused Tasks", show_header=True, header_style="bold orange1")
            paused_table.add_column("Task ID", style="white")

            for task_id in summary["active_paused_ids"]:
                paused_table.add_row(task_id)

            console.print(paused_table)
        else:
            console.print("⏸️ No active paused tasks found", style="dim")

        print_timing("tasks", elapsed_time)

    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error listing tasks: {e}", style="red")
        print_timing("tasks", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project"
)
def jobs(project_name: str):
    """List job counts and sample job IDs for a project."""
    start_time = time.time()

    try:
        summary = list_jobs_summary(project_name=project_name)

        elapsed_time = time.time() - start_time

        # Create summary table
        summary_table = Table(title=f"Job Summary for {project_name}", show_header=True, header_style="bold magenta")
        summary_table.add_column("Status", style="cyan", no_wrap=True)
        summary_table.add_column("Count", style="yellow", justify="right")

        summary_table.add_row("Pending Ingestion", str(summary["pending_ingestion_count"]))
        summary_table.add_row("Ingested", str(summary["ingested_count"]))
        summary_table.add_row("Completed", str(summary["completed_count"]))

        console.print(summary_table)

        # Create pending ingestion jobs table
        if summary["pending_ingestion_ids"]:
            pending_table = Table(title="First 5 Pending Ingestion Jobs", show_header=True, header_style="bold red")
            pending_table.add_column("Job ID", style="white")

            for job_id in summary["pending_ingestion_ids"]:
                pending_table.add_row(job_id)

            console.print(pending_table)
        else:
            console.print("📋 No pending ingestion jobs found", style="dim")

        # Create ingested jobs table
        if summary["ingested_ids"]:
            ingested_table = Table(title="First 5 Ingested Jobs", show_header=True, header_style="bold yellow")
            ingested_table.add_column("Job ID", style="white")

            for job_id in summary["ingested_ids"]:
                ingested_table.add_row(job_id)

            console.print(ingested_table)
        else:
            console.print("📝 No ingested jobs found", style="dim")


        print_timing("jobs", elapsed_time)

    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error listing jobs: {e}", style="red")
        print_timing("jobs", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project"
)
@click.option(
    "--task_id", "-t",
    required=True,
    help="ID of the task to reactivate"
)
def reactivate(project_name: str, task_id: str):
    """Reactivate a completed task by clearing its completion status."""
    start_time = time.time()

    try:
        success = reactivate_task(
            project_name=project_name,
            task_id=task_id
        )

        elapsed_time = time.time() - start_time

        if success:
            console.print(f"✅ Successfully reactivated task: {task_id}", style="green")

            # Get and display updated task details
            task_details = get_task(
                project_name=project_name,
                task_id=task_id
            )
            print_task_details(task_details)
        else:
            console.print(f"❌ Failed to reactivate task: {task_id}", style="red")

        print_timing("reactivate_task", elapsed_time)

    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error reactivating task: {e}", style="red")
        print_timing("reactivate_task", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project to clear"
)
@click.option(
    "--include-users", "-u",
    is_flag=True,
    help="Also clear users"
)
@click.option(
    "--include-task-types", "-t",
    is_flag=True,
    help="Also clear task types"
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Skip confirmation prompt"
)
def clear(project_name: str, include_users: bool, include_task_types: bool, force: bool):
    """Clear all tasks, jobs, and timesheets from a project.
    
    WARNING: This is a destructive operation that cannot be undone!
    """
    start_time = time.time()

    # First, get counts of what will be deleted
    try:
        # Get task and job counts
        # Need to count ALL tasks, including inactive ones
        from zetta_utils.task_management.db.models import TaskModel, JobModel
        from zetta_utils.task_management.db.session import get_session_context
        from sqlalchemy import select, func
        
        with get_session_context() as session:
            # Count ALL tasks (active and inactive)
            total_tasks = session.execute(
                select(func.count()).select_from(TaskModel).where(TaskModel.project_name == project_name)
            ).scalar_one()
            
            # Count all jobs
            total_jobs = session.execute(
                select(func.count()).select_from(JobModel).where(JobModel.project_name == project_name)
            ).scalar_one()
            
            # Get breakdown of tasks by active/inactive
            active_tasks = session.execute(
                select(func.count()).select_from(TaskModel)
                .where(TaskModel.project_name == project_name)
                .where(TaskModel.is_active == True)
            ).scalar_one()
            
            inactive_tasks = total_tasks - active_tasks
        
        # Show what will be deleted with counts
        console.print(f"\n[bold red]⚠️  WARNING: Destructive Operation![/bold red]")
        console.print(f"\nThis will permanently delete from project '{project_name}':")
        console.print(f"  • [yellow]{total_tasks}[/yellow] tasks")
        console.print(f"  • [yellow]{total_jobs}[/yellow] jobs") 
        console.print("  • All dependencies")
        console.print("  • All timesheet entries")
        
        if include_users:
            console.print("  • All users")
        if include_task_types:
            console.print("  • All task types")
        
        # Show breakdown if there are items
        if total_tasks > 0:
            console.print(f"\n[dim]Task breakdown: {active_tasks} active, {inactive_tasks} inactive[/dim]")
        if total_jobs > 0:
            console.print(f"[dim]Job breakdown: {total_jobs} total[/dim]")
        
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not fetch counts: {e}[/yellow]")
        console.print("Proceeding without count information...")
    
    # Confirmation prompt - always require typing project name unless --force
    if not force:
        console.print(f"\n[bold yellow]To confirm deletion, type the project name: [red]{project_name}[/red][/bold yellow]")
        confirmation = input("> ")
        
        if confirmation != project_name:
            console.print("[red]❌ Confirmation failed. Operation cancelled.[/red]")
            return
    
    try:
        # Clear main data (tasks, jobs, dependencies, timesheets)
        console.print("\n[cyan]Clearing project data...[/cyan]")
        deleted_counts = clear_project_data(project_name=project_name)
        
        # Display results
        table = Table(title="Deleted Records", show_header=True, header_style="bold red")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="yellow", justify="right")
        
        table.add_row("Timesheets", str(deleted_counts["timesheets"]))
        table.add_row("Dependencies", str(deleted_counts["dependencies"]))
        table.add_row("Tasks", str(deleted_counts["tasks"]))
        table.add_row("Jobs", str(deleted_counts["jobs"]))
        
        # Clear users if requested
        if include_users:
            user_count = clear_project_users(project_name=project_name)
            table.add_row("Users", str(user_count))
        
        # Clear task types if requested
        if include_task_types:
            task_type_count = clear_project_task_types(project_name=project_name)
            table.add_row("Task Types", str(task_type_count))
        
        console.print(table)
        
        elapsed_time = time.time() - start_time
        console.print(f"\n✅ Successfully cleared project '{project_name}'", style="green")
        print_timing("clear_project", elapsed_time)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error clearing project: {e}", style="red")
        print_timing("clear_project", elapsed_time)
        raise click.ClickException(str(e))


@task_mgmt.command()
@click.option(
    "--project_name", "-p",
    required=True,
    help="Name of the project"
)
@click.option(
    "--seed_sv_id", "-s",
    required=True,
    type=int,
    help="Seed supervoxel ID"
)
def segment_link(project_name: str, seed_sv_id: int):
    """Generate neuroglancer link for a segment by seed supervoxel ID."""
    start_time = time.time()
    
    try:
        link = get_segment_link(project_name=project_name, seed_sv_id=seed_sv_id)
        elapsed_time = time.time() - start_time
        
        console.print(f"✅ Generated link for segment with seed_sv_id {seed_sv_id}", style="green")
        console.print(f"\n🔗 Neuroglancer Link:", style="bold cyan")
        console.print(link, style="blue underline")
        
        print_timing("segment_link", elapsed_time)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        console.print(f"❌ Error generating segment link: {e}", style="red")
        print_timing("segment_link", elapsed_time)
        raise click.ClickException(str(e))


# Export the command group
__all__ = ["task_mgmt"]
