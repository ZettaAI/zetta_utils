"""
Segmentation Auto Verifier Worker

An automated worker that polls for tasks, processes them by parsing the ng_state,
and automatically releases them with "pass" status.
"""

import json
import time

import attrs
import os
import click
import numpy as np
from cloudvolume import CloudVolume
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text
from caveclient import CAVEclient
import pcg_skel
import meshparty

from zetta_utils import log
from zetta_utils.task_management.task import get_task, release_task, start_task

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

slack_client = WebClient(token=os.environ["ZETTA_PROOFREDING_BOT_SLACK_TOKEN"])

logger = log.get_logger()
console = Console()


@attrs.mutable
class SegmentSkeleton:
    """Class to hold segment ID and skeleton data."""
    segment_id: int = attrs.field()
    skeleton: meshparty.skeleton.Skeleton = attrs.field()

def get_skeleton(ng_dict: dict) -> SegmentSkeleton | None:
    """Get skeleton from segmentation layer."""
    
    for layer in ng_dict["layers"]:
        if layer["name"] == "Segmentation":
            source = layer["source"]
            segments = layer["segments"]
            
            if source and segments:
                segment_id = int(segments[0])
                client = CAVEclient(
                    datastack_name="kronauer_ant", 
                    server_address="https://proofreading.zetta.ai",
                    auth_token_file="~/.cloudvolume/secrets/cave-secret.json"   
                )
                skel = pcg_skel.pcg_skeleton(root_id=segment_id, client=client)
                pathlength = skel.path_length()
                cv = CloudVolume(source)
                skeleton = cv.skeleton.get(segment_id)
                
                return SegmentSkeleton(segment_id=segment_id, skeleton=skeleton)
    return None


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
def run_worker(user_id: str, project_name: str, polling_period: float, min_skeleton_length_mm: int, slack_channel: str):
    """Run the segmentation auto verifier worker."""
    # Display startup banner
    startup_panel = Panel(
        f"[bold cyan]Segmentation Auto Verifier Worker[/bold cyan]\n\n"
        f"[yellow]User ID:[/yellow] {user_id}\n"
        f"[yellow]Project:[/yellow] {project_name}\n"
        f"[yellow]Polling Period:[/yellow] {polling_period}s\n"
        f"[yellow]Min Skeleton Length:[/yellow] {min_skeleton_length_mm}mm\n\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        title="🤖 Worker Starting",
        border_style="blue"
    )
    console.print(startup_panel)
    
    logger.info(f"Starting segmentation auto verifier worker for project '{project_name}' with user '{user_id}'")
    
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
                    
                    # Create processing panel
                    processing_text = Text(f"Processing Task #{task_count}", style="bold green")
                    processing_panel = Panel(
                        f"[cyan]Task ID:[/cyan] {task_id}\n"
                        f"[cyan]Status:[/cyan] Analyzing ng_state...",
                        title=processing_text,
                        border_style="green"
                    )
                    console.print(processing_panel)
                    
                    # Get and parse ng_state
                    task_details = get_task(project_name=project_name, task_id=task_id)
                    task_type = task_details.get("task_type", "unknown")
                    
                    logger.info(f"Processing task {task_id} (type: {task_type})")
                    console.print(f"[cyan]Task Type:[/cyan] {task_type}")
                    
                    ng_state = task_details.get("ng_state", "")
                    logger.info(f"Successfully parsed ng_state for task {task_id}")
                    
                    slack_client.chat_postMessage(channel=slack_channel, text="") 
                    # Get skeleton from segmentation layer
                    breakpoint()
                    skeleton_info = get_skeleton(ng_state)
                    if skeleton_info:
                        # Convert skeleton length from nm to mm (1 mm = 1,000,000 nm)
                        skeleton_length_nm = skeleton_info.skeleton.path_length()
                        skeleton_length_mm = skeleton_length_nm / 1_000_000
                        
                        console.print(f"[green]✅ Got skeleton for segment {skeleton_info.segment_id}[/green]")
                        console.print(f"[cyan]Skeleton length:[/cyan] {skeleton_length_mm:.2f}mm ({skeleton_length_nm:.0f}nm)")
                        console.print(f"[cyan]Minimum required:[/cyan] {min_skeleton_length_mm}mm")
                        
                        if skeleton_length_mm >= min_skeleton_length_mm:
                            console.print(f"[green]✅ Skeleton meets minimum length requirement[/green]")
                        else:
                            console.print(f"[red]❌ Skeleton below minimum length requirement[/red]")
                    # Release task with "pass" status
                    logger.info(f"Releasing task {task_id} with 'pass' status")
                    
                    #success = release_task(
                    #    project_name=project_name,
                    #    task_id=task_id,
                    #    user_id=user_id,
                    #    completion_status="pass"
                    #)
                    
                    #if success:
                    #    logger.info(f"Successfully released task {task_id}")
                    #    console.print(f"[green]✅ Released task {task_id} with status: [bold]pass[/bold][/green]\n")
                    #else:
                    #    logger.error(f"Failed to release task {task_id}")
                    #    console.print(f"[red]❌ Failed to release task {task_id}[/red]\n")
                    
                    # had_task_last_time = True
                    
                else:
                    if had_task_last_time:  # Only log when transitioning from having tasks to no tasks
                        logger.info("No tasks available, entering wait mode")
                    console.print("[dim]⏳ No tasks available, waiting...[/dim]")
                    had_task_last_time = False
                    
            except Exception as e:
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
    run_worker()