"""
PyChunkedGraph edit event listener worker.

Listens to Pub/Sub messages from PyChunkedGraph and updates supervoxels table
when segment merges or splits occur.
"""

import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests
from caveclient import CAVEclient

from zetta_utils.message_queues.pubsub import PubSubPullQueue
from zetta_utils.task_management.db.models import (
    ProjectModel,
    SegmentModel,
    SupervoxelModel,
)
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment_queue import queue_segment_updates_for_segments
from zetta_utils.task_management.supervoxel import (
    get_supervoxels_by_segment,
    update_supervoxels_for_merge,
    update_supervoxels_for_split,
)

logger = logging.getLogger(__name__)


def _find_moved_seed_ids(old_root_ids: list[int]) -> list[int]:
    """Return seed_ids that currently belong to any of old_root_ids roots.

    Looks up SegmentModel.seed_id by joining SupervoxelModel on supervoxel_id.
    """
    if not old_root_ids:
        return []

    try:
        with get_session_context() as session:
            rows = (
                session.query(SegmentModel.seed_id)
                .join(
                    SupervoxelModel, SupervoxelModel.supervoxel_id == SegmentModel.seed_id
                )
                .filter(SupervoxelModel.current_segment_id.in_(old_root_ids))
                .all()
            )
            return [int(r.seed_id) for r in rows]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning(f"Could not pre-compute moved seed_ids: {exc}")
        return []


def _resolve_duplicates_in_root(
    project_name: str, root_id: int
) -> int:
    """Mark all but one segment under a root as Duplicate.

    Keeps the primary seed (earliest last_merge_at; None treated as earliest).
    Tie-breakers: created_at, then seed_id.

    Returns number of segments marked as Duplicate.
    """
    with get_session_context() as session:
        rows = (
            session.query(
                SegmentModel.seed_id, SegmentModel.last_merge_at, SegmentModel.created_at
            )
            .filter(
                SegmentModel.project_name == project_name,
                SegmentModel.current_segment_id == root_id,
                SegmentModel.status != "Duplicate",
            )
            .all()
        )

        if not rows:
            return 0

        def sort_key(r):
            lm = r.last_merge_at.timestamp() if r.last_merge_at else float("-inf")
            created = r.created_at.timestamp() if r.created_at else 0.0
            return (lm, created, int(r.seed_id))

        rows.sort(key=sort_key)  # earliest first
        primary_seed_id = int(rows[0].seed_id)

        now_ts = datetime.now(timezone.utc)
        result = (
            session.query(SegmentModel)
            .filter(
                SegmentModel.project_name == project_name,
                SegmentModel.current_segment_id == root_id,
                SegmentModel.status != "Duplicate",
                SegmentModel.seed_id != primary_seed_id,
            )
            .update(
                {SegmentModel.status: "Duplicate", SegmentModel.updated_at: now_ts},
                synchronize_session=False,
            )
        )
        session.commit()
        return int(result)


def _apply_merge_updates(
    project_name: str,
    moved_seed_ids: list[int],
    new_root_id: int,
    edit_timestamp: datetime,
) -> None:
    """Update moved segments and resolve duplicates under the new root.

    - Sets current_segment_id and last_merge_at for moved seeds
    - Resolves duplicates for the new_root_id
    """
    if not moved_seed_ids:
        # Nothing moved — still run duplicate resolution in case other seeds collide
        _resolve_duplicates_in_root(project_name, new_root_id)
        return

    with get_session_context() as session:
        now_ts = datetime.now(timezone.utc)
        (
            session.query(SegmentModel)
            .filter(
                SegmentModel.project_name == project_name,
                SegmentModel.seed_id.in_(moved_seed_ids),
            )
            .update(
                {
                    SegmentModel.current_segment_id: new_root_id,
                    SegmentModel.last_merge_at: edit_timestamp,
                    SegmentModel.updated_at: now_ts,
                },
                synchronize_session=False,
            )
        )
        session.commit()

    _resolve_duplicates_in_root(project_name, new_root_id)


def get_old_roots_from_lineage_graph(
    server_address: str,
    table_id: str,
    root_ids: list[int],
    timestamp_past: datetime,
) -> dict[int, list[int]]:
    """
    Get old root IDs from PyChunkedGraph lineage_graph endpoint.

    Args:
        server_address: PCG server address
        table_id: Table/graph ID
        root_ids: List of new root IDs to query
        timestamp_past: Timestamp to limit past edges

    Returns:
        Dict mapping new_root_id -> list of old_root_ids
    """
    try:
        # PCG lineage_graph_multiple endpoint
        url = f"{server_address}/segmentation/api/v1/table/{table_id}/lineage_graph_multiple"

        # Convert timestamp to milliseconds since epoch
        timestamp_ms = int(timestamp_past.timestamp() * 1000)

        payload = {
            "root_ids": root_ids,
            "timestamp_past": timestamp_ms,
        }

        print(f"[DEBUG] Lineage graph request URL: {url}")
        print(f"[DEBUG] Lineage graph request payload: {payload}")

        # Get auth token from environment
        auth_token = os.getenv("CAVE_AUTH_TOKEN")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            print("[DEBUG] Using auth token from CAVE_AUTH_TOKEN")
        else:
            print("[DEBUG] No auth token found in CAVE_AUTH_TOKEN env var")

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"[DEBUG] Response status code: {response.status_code}")
        print(f"[DEBUG] Response headers: {response.headers}")
        print(f"[DEBUG] Response text (first 1000 chars): {response.text[:1000]}")
        response.raise_for_status()

        data = response.json()
        print(f"[DEBUG] Lineage graph raw response (first 500 chars): {str(data)[:500]}")
        logger.info(f"Got lineage graph for {len(root_ids)} roots")

        # Parse lineage graph to extract old roots
        # Response format: {"links": [{"source": <old_root>, "target": <new_root>}, ...]}
        result = {}
        links = data.get("links", [])
        print(f"[DEBUG] Found {len(links)} total links in lineage graph")

        for root_id in root_ids:
            old_roots = set()
            # Find all edges where target is our new root
            for link in links:
                source = link.get("source")
                target = link.get("target")
                # If target is our new root, source is an old root (immediate parent)
                if target == root_id:
                    old_roots.add(source)
                    print(f"[DEBUG] Found edge: {source} -> {target}")

            if old_roots:
                result[root_id] = list(old_roots)
                logger.info(f"Found {len(old_roots)} old roots for {root_id}: {old_roots}")
                print(f"[DEBUG] Old roots for {root_id}: {old_roots}")
            else:
                print(f"[DEBUG] No old roots found for {root_id}")

        return result
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Failed to get lineage graph from PCG: {e}")
        print(f"[DEBUG] Exception details: {e}")
        traceback.print_exc()
        return {}


def get_old_roots_from_lineage(
    cave_client: CAVEclient,
    new_root_id: int,
    timestamp_past: Optional[datetime] = None,
) -> list[int]:
    """
    Get old root IDs that were merged into the new root using lineage graph.

    Args:
        cave_client: CAVE client instance
        new_root_id: New root ID to query
        timestamp_past: How far back to look (None = from beginning)

    Returns:
        List of old root IDs that merged into this new root
    """
    try:
        timestamp_past_ms = None
        if timestamp_past:
            timestamp_past_ms = int(timestamp_past.timestamp() * 1000)

        lineage = cave_client.chunkedgraph.get_lineage_graph(
            root_id=new_root_id,
            timestamp_past=timestamp_past_ms,
        )

        old_roots = []
        for node_id in lineage["nodes"]:
            if node_id != str(new_root_id):
                old_roots.append(int(node_id))

        return old_roots
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error getting lineage for root {new_root_id}: {e}")
        return []


def get_supervoxel_ids_from_segment(
    server_address: str,
    table_id: str,
    segment_id: int,
) -> list[int]:
    """
    Get supervoxel IDs (leaves) for a given segment ID using PyChunkedGraph API.

    Args:
        server_address: PCG server address
        table_id: Table/graph ID
        segment_id: Segment/root ID to get supervoxels for

    Returns:
        List of supervoxel IDs for the segment
    """
    try:
        # PCG leaves endpoint - gets all supervoxel IDs for a segment
        url = (
            f"{server_address}/segmentation/api/v1/table/{table_id}/"
            f"node/{segment_id}/leaves"
        )

        # Get auth token from environment
        auth_token = os.getenv("CAVE_AUTH_TOKEN")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        print(f"[DEBUG] Getting supervoxels for segment {segment_id}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        supervoxel_ids = data.get("leaf_ids", [])

        print(
            f"[DEBUG] Got {len(supervoxel_ids)} supervoxels "
            f"for segment {segment_id}"
        )
        return [int(sv_id) for sv_id in supervoxel_ids]

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error getting supervoxels for segment {segment_id}: {e}")
        print(f"[DEBUG] Error getting supervoxels for segment {segment_id}: {e}")
        return []


def get_root_for_coordinate_pcg(
    server_address: str,
    table_id: str,
    x: float,
    y: float,
    z: float,
) -> Optional[int]:
    """
    Get the current root ID for a given coordinate using PyChunkedGraph API.

    Args:
        server_address: PCG server address
        table_id: Table/graph ID
        x: X coordinate (in nm)
        y: Y coordinate (in nm)
        z: Z coordinate (in nm)

    Returns:
        Root ID at the coordinate, or None if failed
    """
    try:
        # PCG node_id endpoint - queries root at a given coordinate
        url = f"{server_address}/segmentation/api/v1/table/{table_id}/node_id"

        # Coordinates should be in nm (same units as stored in supervoxel seeds)
        payload = {
            "x": x,
            "y": y,
            "z": z,
        }

        # Get auth token from environment
        auth_token = os.getenv("CAVE_AUTH_TOKEN")
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        print(f"[DEBUG] Querying root at coordinate ({x}, {y}, {z})")
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        print(f"[DEBUG] Got response for ({x}, {y}, {z}): {data}")

        # Response format varies - might be {"node_id": <id>} or {"root_id": <id>}
        root_id = data.get("node_id") or data.get("root_id")
        if root_id:
            return int(root_id)

        # Sometimes response is just the ID directly
        if isinstance(data, (int, str)):
            return int(data)

        return None
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error getting root for coordinate ({x}, {y}, {z}): {e}")
        print(f"[DEBUG] Full error for ({x}, {y}, {z}): {e}")
        return None


def get_new_root_for_coordinate(
    cave_client: CAVEclient,
    x: float,
    y: float,
    z: float,
) -> Optional[int]:
    """
    Get the current root ID for a given coordinate using CAVEclient.

    Args:
        cave_client: CAVE client instance
        x: X coordinate
        y: Y coordinate
        z: Z coordinate

    Returns:
        Current root ID at that coordinate, or None if error
    """
    try:
        root_id = cave_client.chunkedgraph.get_root_id(
            supervoxel_id=None,
            location=[x, y, z],
        )
        return int(root_id) if root_id else None
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error getting root for coordinate ({x}, {y}, {z}): {e}")
        return None


def process_merge_event(
    event_data: dict[str, Any],
    project_name: str,
    cave_client: Optional[CAVEclient],
    server_address: str,
    table_id: str,
) -> None:
    """
    Process a merge event from PyChunkedGraph.

    Args:
        event_data: Event data from Pub/Sub
        project_name: Project name
        cave_client: CAVE client instance (optional)
        server_address: PCG server address
        table_id: Table/graph ID from message attributes
    """
    try:
        # PyChunkedGraph sends 'new_root_ids' (plural) as a list
        new_root_ids = event_data.get("new_root_ids", [])
        new_root_id = new_root_ids[0] if new_root_ids else None

        # Use current time if no timestamp provided
        edit_timestamp = datetime.now(timezone.utc)
        if "timestamp" in event_data:
            ts = event_data["timestamp"]
            if isinstance(ts, str):
                edit_timestamp = datetime.fromisoformat(ts)
            elif isinstance(ts, (int, float)):
                edit_timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)

        timestamp_int = int(edit_timestamp.timestamp())
        event_id = str(
            event_data.get("operation_id", f"{new_root_id}_{timestamp_int}")
        )

        if not new_root_id:
            logger.warning("No new_root_id in event data")
            return

        old_roots = event_data.get("old_root_ids")
        print(
            f"[DEBUG] Processing merge - operation_id: {event_id}, "
            f"new_root_id: {new_root_id}, old_roots from event: {old_roots}"
        )

        # Get old roots from PyChunkedGraph lineage_graph if not in event
        if not old_roots:
            print(f"[DEBUG] Querying lineage_graph for new_root_ids: {new_root_ids}")
            # Query lineage for all new roots (usually just one for merges)
            # Use timestamp 1 minute before current time
            timestamp_past = datetime.now(timezone.utc) - timedelta(minutes=1)
            timestamp_past_ms = int(timestamp_past.timestamp() * 1000)
            print(
                f"[DEBUG] Using timestamp_past: {timestamp_past} "
                f"({timestamp_past_ms} ms)"
            )

            lineage_results = get_old_roots_from_lineage_graph(
                server_address=server_address,
                table_id=table_id,
                root_ids=new_root_ids,
                timestamp_past=timestamp_past,
            )
            print(f"[DEBUG] Lineage graph results: {lineage_results}")

            # Get old roots for the primary new root
            if new_root_id in lineage_results:
                old_roots = lineage_results[new_root_id]
                logger.info(f"Got {len(old_roots)} old roots from lineage graph")

        # Fall back to CAVEclient lineage query (if available)
        if not old_roots and cave_client is not None:
            print("[DEBUG] Attempting to get old roots from CAVEclient lineage")
            old_roots = get_old_roots_from_lineage(
                cave_client=cave_client,
                new_root_id=new_root_id,
                timestamp_past=edit_timestamp,
            )
            print(f"[DEBUG] CAVEclient lineage returned: {old_roots}")

        if not old_roots:
            logger.warning(
                f"No old_root_ids found for operation {event_id} (table: {table_id}). "
                "Cannot determine which segments merged. Skipping update."
            )
            return

        if old_roots:
            count = update_supervoxels_for_merge(
                old_root_ids=old_roots,
                new_root_id=new_root_id,
                project_name=project_name,
                event_id=event_id,
                edit_timestamp=edit_timestamp,
                operation_type="merge",
            )
            print(f"[DEBUG] Updated {count} supervoxels for merge")
            logger.info(
                f"Updated {count} supervoxels for merge: {old_roots} -> {new_root_id}"
            )

            # Queue segment updates for affected segments
            try:
                all_affected_segments = old_roots + [new_root_id]
                queued_count = queue_segment_updates_for_segments(
                    project_name=project_name,
                    segment_ids=all_affected_segments
                )
                print(f"[DEBUG] Queued {queued_count} segment updates for merge")
                logger.info(f"Queued {queued_count} segment updates for merge")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[DEBUG] Error queueing segment updates for merge: {e}")
                logger.error(f"Failed to queue segment updates for merge: {e}")

            # Update moved segments and resolve duplicates
            moved_seed_ids = _find_moved_seed_ids(old_roots)
            try:
                _apply_merge_updates(
                    project_name=project_name,
                    moved_seed_ids=moved_seed_ids,
                    new_root_id=new_root_id,
                    edit_timestamp=edit_timestamp,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error(f"Error updating segments/duplicates after merge: {exc}")
        else:
            logger.warning(f"No old roots found for new root {new_root_id}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error processing merge event: {e}", exc_info=True)


def process_split_event(  # pylint: disable=unused-argument
    event_data: dict[str, Any],
    project_name: str,
    cave_client: Optional[CAVEclient],
    server_address: str,
    table_id: str,
) -> None:
    """
    Process a split event from PyChunkedGraph.

    For splits, we get supervoxel IDs directly from each new root segment
    using the PyChunkedGraph leaves endpoint.

    Args:
        event_data: Event data from Pub/Sub
        project_name: Project name
        cave_client: CAVE client instance (optional, not currently used for splits)
        server_address: PCG server address
        table_id: Table/graph ID
    """
    try:
        # PyChunkedGraph sends multiple new_root_ids for splits
        new_root_ids = event_data.get("new_root_ids", [])

        # Use current time if no timestamp provided
        edit_timestamp = datetime.now(timezone.utc)
        if "timestamp" in event_data:
            ts = event_data["timestamp"]
            if isinstance(ts, str):
                edit_timestamp = datetime.fromisoformat(ts)
            elif isinstance(ts, (int, float)):
                edit_timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)

        timestamp_int = int(edit_timestamp.timestamp())
        event_id = str(event_data.get("operation_id", f"split_{timestamp_int}"))
        print(event_data)

        if len(new_root_ids) < 2:
            logger.warning(
                f"Split event should have multiple new_root_ids, "
                f"got {len(new_root_ids)}"
            )
            return

        print(
            f"[DEBUG] Processing split - event_id: {event_id}, "
            f"new_root_ids: {new_root_ids}"
        )

        # Query lineage graph to find the old root
        timestamp_past = datetime.now(timezone.utc) - timedelta(minutes=1)
        lineage_results = get_old_roots_from_lineage_graph(
            server_address=server_address,
            table_id=table_id,
            root_ids=new_root_ids,
            timestamp_past=timestamp_past,
        )

        # Find the common old root (should be same for all new roots in a split)
        old_root_id = None
        for new_root in new_root_ids:
            if new_root in lineage_results and lineage_results[new_root]:
                old_root_id = lineage_results[new_root][0]
                break

        if not old_root_id:
            logger.warning(f"Could not find old root for split event {event_id}")
            return

        print(f"[DEBUG] Found old root {old_root_id} for split into {new_root_ids}")

        # Get all supervoxels that currently belong to the old root
        old_supervoxels = get_supervoxels_by_segment(segment_id=old_root_id)
        print(
            f"[DEBUG] Found {len(old_supervoxels)} supervoxels currently "
            f"assigned to old root {old_root_id}"
        )

        # Build supervoxel assignments mapping
        supervoxel_assignments = {}
        for new_root_id in new_root_ids:
            # Get all supervoxel IDs that belong to this new root
            new_segment_supervoxel_ids = get_supervoxel_ids_from_segment(
                server_address=server_address,
                table_id=table_id,
                segment_id=new_root_id,
            )

            print(
                f"[DEBUG] Got {len(new_segment_supervoxel_ids)} supervoxels "
                f"for new root {new_root_id}"
            )

            # Convert to set for faster lookup
            new_segment_supervoxel_set = set(new_segment_supervoxel_ids)

            # Check each supervoxel from old root
            for supervoxel in old_supervoxels:
                if supervoxel.supervoxel_id in new_segment_supervoxel_set:
                    supervoxel_assignments[supervoxel.supervoxel_id] = new_root_id
                    print(
                        f"[DEBUG] Assigning supervoxel "
                        f"{supervoxel.supervoxel_id}: "
                        f"{old_root_id} -> {new_root_id}"
                    )

        # Update all supervoxels in a single transaction with event tracking
        updated_count = update_supervoxels_for_split(
            old_root_id=old_root_id,
            new_root_ids=new_root_ids,
            supervoxel_assignments=supervoxel_assignments,
            project_name=project_name,
            event_id=event_id,
            edit_timestamp=edit_timestamp,
            operation_type="split",
        )

        logger.info(
            f"Completed split processing for root {old_root_id}: "
            f"updated {updated_count} supervoxels across {len(new_root_ids)} new roots"
        )

        # Queue segment updates for affected segments
        try:
            all_affected_segments = [old_root_id] + new_root_ids
            queued_count = queue_segment_updates_for_segments(
                project_name=project_name,
                segment_ids=all_affected_segments
            )
            logger.info(f"Queued {queued_count} segment updates for split")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Failed to queue segment updates for split: {e}")

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error(f"Error processing split event: {e}", exc_info=True)


def process_edit_event(
    event_data: dict[str, Any],
    message_attributes: dict[str, str],
    project_name: str,
    cave_client: CAVEclient,
    server_address: str,
) -> None:
    """
    Process an edit event from PyChunkedGraph.

    Args:
        event_data: Event data from Pub/Sub
        message_attributes: Message attributes from Pub/Sub
        project_name: Project name
        cave_client: CAVE client instance
        server_address: PCG server address
    """
    table_id = message_attributes.get("table_id", project_name)

    # Detect operation type from new_root_ids count
    new_root_ids = event_data.get("new_root_ids", [])

    if len(new_root_ids) == 1:
        # Single new root = merge
        process_merge_event(
            event_data, project_name, cave_client, server_address, table_id
        )
    elif len(new_root_ids) > 1:
        # Multiple new roots = split
        print(f"[DEBUG] Detected split: {len(new_root_ids)} new roots")
        process_split_event(
            event_data, project_name, cave_client, server_address, table_id
        )
    else:
        logger.warning("No new_root_ids in event data")


def run_pcg_edit_listener(  # pylint: disable=too-many-branches,too-many-statements
    project_id: str,
    subscription_name: str,
    project_name: Optional[str] = None,
    datastack_name: Optional[str] = None,
    server_address: Optional[str] = None,
    poll_interval_sec: int = 5,
    max_messages: int = 10,
) -> None:
    """
    Run the PCG edit listener worker.

    Args:
        project_id: GCP project ID
        subscription_name: Pub/Sub subscription name
        project_name: Task management project name (auto-detected from
            table_id in messages if not provided)
        datastack_name: CAVE datastack name (auto-detected from project)
        server_address: PCG server address (auto-detected from project,
            defaults to https://data.proofreading.zetta.ai)
        poll_interval_sec: How often to poll for messages
        max_messages: Maximum messages to pull per batch
    """
    print("[DEBUG] Starting PCG edit listener")
    logger.info("Starting PCG edit listener")

    # Get project configuration from database if project_name provided
    project = None
    if project_name:
        print("[DEBUG] Connecting to database...")
        with get_session_context() as session:
            print("[DEBUG] Database connection established")
            print(f"[DEBUG] Querying for project: {project_name}")
            project = (
                session.query(ProjectModel)
                .filter_by(project_name=project_name)
                .first()
            )
            print(f"[DEBUG] Project found: {project is not None}")
            if not project:
                raise ValueError(f"Project '{project_name}' not found!")

        print("[DEBUG] Checking datastack_name...")
        # Use provided values or get from project
        if datastack_name is None and project:
            datastack_name = project.datastack_name
            if not datastack_name:
                raise ValueError(
                    f"No datastack_name for project '{project_name}'"
                )
        print(f"[DEBUG] Datastack: {datastack_name}")

    # Set server_address (default if not provided)
    print("[DEBUG] Checking server_address...")
    if server_address is None:
        # Follow same pattern as cave_synapse_mgr.py
        if datastack_name and datastack_name.startswith("wclee"):
            server_address = "https://global.daf-apis.com"
        else:
            server_address = "https://data.proofreading.zetta.ai"
    print(f"[DEBUG] Server: {server_address}")

    print("[DEBUG] Creating PubSub queue...")
    logger.info(f"Using datastack: {datastack_name}")
    logger.info(f"Using server: {server_address}")

    pubsub_queue: PubSubPullQueue = PubSubPullQueue(
        name="pcg_edit_queue",
        project_id=project_id,
        subscription_name=subscription_name,
    )
    print("[DEBUG] PubSub queue created")

    # Try to create CAVE client for lineage queries
    # Optional: works without it if PubSub events include old_root_ids
    print("[DEBUG] Initializing CAVEclient...")
    cave_client = None
    try:
        cave_client = CAVEclient(
            datastack_name=datastack_name,
            server_address=server_address,
            auth_token=os.getenv("CAVE_AUTH_TOKEN"),
        )
        logger.info("CAVEclient initialized successfully")
        print("[DEBUG] CAVEclient initialized successfully")
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(f"Could not initialize CAVEclient: {e}")
        logger.info(
            "Continuing without CAVEclient. "
            "Events must include old_root_ids."
        )
        print(f"[DEBUG] CAVEclient failed: {e}")

    print("[DEBUG] Starting main listener loop...")
    logger.info(
        "Listening to subscription: "
        f"projects/{project_id}/subscriptions/{subscription_name}"
    )
    print(
        "[DEBUG] Waiting for messages from PubSub "
        "(this will block until messages arrive)..."
    )

    while True:
        try:
            messages = pubsub_queue.pull(max_num=max_messages)
            print(f"[DEBUG] Received {len(messages)} message(s)")

            for msg in messages:
                try:
                    # Extract attributes from payload
                    message_attributes = msg.payload.pop("_pubsub_attributes", {})

                    # Use table_id as project_name if project_name not provided
                    msg_project_name = (
                        project_name or message_attributes.get("table_id")
                    )
                    if not msg_project_name:
                        logger.warning(
                            "No project_name provided and no table_id in "
                            "message attributes. Skipping message."
                        )
                        msg.acknowledge_fn()
                        continue

                    process_edit_event(
                        msg.payload,
                        message_attributes,
                        msg_project_name,
                        cave_client,
                        server_address
                    )
                    msg.acknowledge_fn()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(f"Error processing message: {e}", exc_info=True)

            if not messages:
                time.sleep(poll_interval_sec)

        except KeyboardInterrupt:
            logger.info("Shutting down PCG edit listener")
            break
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(poll_interval_sec)
