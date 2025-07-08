import copy
import uuid
from datetime import datetime, timezone
from typing import Any

from zetta_utils import log

from ..db.models import EndpointModel, ProjectModel, SegmentModel
from ..db.session import get_session_context
from ..task import create_task
from ..types import Task
from ..utils import generate_id_nonunique
from .completion import register_completion_handler
from .creation import register_creation_handler
from .verification import VerificationResult, register_verifier

logger = log.get_logger()


def get_layer_by_name(ng_state: dict, layer_name: str) -> dict | None:
    """Get a layer from neuroglancer state by name."""
    layers = ng_state.get("layers", [])
    for layer in layers:
        if layer.get("name") == layer_name:
            return layer
    return None


def get_visible_segment_id(segmentation_layer: dict) -> str | None:
    """Get the ID of the visible segment (not starting with '!')."""
    segments = segmentation_layer.get("segments", [])
    for segment in segments:
        if isinstance(segment, str) and not segment.startswith("!"):
            return segment
    return None


def count_visible_segments(segmentation_layer: dict) -> int:
    """Count visible segments in a segmentation layer."""
    if not segmentation_layer:
        return 0

    # In Neuroglancer, segments starting with "!" are hidden
    segments = segmentation_layer.get("segments", [])
    visible_count = 0

    for segment in segments:
        if isinstance(segment, str) and not segment.startswith("!"):
            visible_count += 1

    return visible_count


def count_annotations(annotation_layer: dict) -> int:
    """Count annotations in an annotation layer."""
    if not annotation_layer:
        return 0
    return len(annotation_layer.get("annotations", []))


def round_coords(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Round coordinates to 6 decimal places for consistent comparison."""
    return (round(x, 6), round(y, 6), round(z, 6))


def extract_annotations_from_ng_state(ng_state: dict) -> dict[str, Any]:
    """Extract all annotations from neuroglancer state."""
    annotations: dict[str, Any] = {
        "seed": None,
        "root": None,
        "certain_ends": [],
        "uncertain_ends": [],
        "breadcrumbs": [],
    }

    for layer in ng_state.get("layers", []):
        if layer.get("type") == "annotation":
            layer_name = layer.get("name", "")

            for annotation in layer.get("annotations", []):
                if "point" in annotation:
                    point = annotation["point"]

                    if layer_name == "Seed Location" and annotations["seed"] is None:
                        annotations["seed"] = point
                    elif layer_name == "Root Location":
                        annotations["root"] = point
                    elif layer_name == "Certain Ends":
                        annotations["certain_ends"].append(point)
                    elif layer_name == "Uncertain Ends":
                        annotations["uncertain_ends"].append(point)
                    elif layer_name == "Breadcrumbs":
                        annotations["breadcrumbs"].append(point)

    return annotations


def verify_trace_layers(ng_state: dict, require_single_segment: bool = True) -> VerificationResult:
    """Verify that neuroglancer state has required trace layers.

    Args:
        ng_state: The neuroglancer state to verify
        require_single_segment: If True, require exactly 1 visible segment

    Returns:
        VerificationResult indicating pass/fail
    """
    # Check required layers exist
    required_layers = [
        "Segmentation",
        "Breadcrumbs",
        "Certain Ends",
        "Uncertain Ends",
        "Seed Location",
    ]
    missing_layers = []

    for layer_name in required_layers:
        if not get_layer_by_name(ng_state, layer_name):
            missing_layers.append(layer_name)

    if missing_layers:
        return VerificationResult(
            passed=False, message=f"Missing required layers: {', '.join(missing_layers)}"
        )

    # Check visible segments if required
    if require_single_segment:
        segmentation_layer = get_layer_by_name(ng_state, "Segmentation")
        visible_segments = count_visible_segments(segmentation_layer) if segmentation_layer else 0

        if visible_segments != 1:
            return VerificationResult(
                passed=False,
                message=(
                    f"Segmentation layer must have exactly 1 visible segment, "
                    f"found {visible_segments}"
                ),
            )

    return VerificationResult(passed=True, message="Trace layers validation passed")


@register_completion_handler("trace_v0")
def handle_trace_v0_completion(  # pylint: disable=too-many-statements
    project_name: str, task: Task, completion_status: str
) -> None:
    """Handle completion of trace_v0 tasks.

    Updates segment data and saves endpoints to database.
    Assumes verification has already been done.
    """
    # For faulty tasks, skip all processing
    if completion_status == "Faulty Task":
        logger.info(f"Task {task['task_id']} marked as faulty - skipping segment updates")
        return

    ng_state = task["ng_state"]

    # Get the visible segment ID
    segmentation_layer = get_layer_by_name(ng_state, "Segmentation")
    segment_id = get_visible_segment_id(segmentation_layer) if segmentation_layer else None

    # Extract all annotations from the state
    annotations = extract_annotations_from_ng_state(ng_state)

    # Get task and user info
    task_id = task["task_id"]
    extra_data = task.get("extra_data", {})
    if not extra_data:
        logger.error(f"Task {task_id} missing extra_data")
        return
    seed_id = extra_data.get("seed_id")
    if not seed_id:
        logger.error(f"Task {task_id} missing seed_id in extra_data")
        return
    user_id = task.get("completed_user_id", task["active_user_id"])

    with get_session_context() as session:
        # Get segment
        segment = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .first()
        )

        if not segment:
            raise ValueError(f"Segment {seed_id} not found in project {project_name}")

        # Update segment fields
        segment.current_segment_id = int(segment_id) if segment_id else None

        if annotations["root"]:
            segment.root_x = annotations["root"][0]
            segment.root_y = annotations["root"][1]
            segment.root_z = annotations["root"][2] - 0.5  # Subtract 0.5 from z for root location

        if task_id not in segment.task_ids:
            segment.task_ids = segment.task_ids + [task_id]

        # Set status based on completion status
        if completion_status == "Done":
            segment.status = "Completed"
        elif completion_status in ["Merger", "Wrong Cell Type"]:
            segment.status = "Abandoned"
        else:
            assert completion_status == "Can't Continue"
            segment.status = "WIP"

        segment.updated_at = datetime.now(timezone.utc)

        # Delete all existing endpoints for this seed_id
        deleted_count = (
            session.query(EndpointModel)
            .filter_by(project_name=project_name, seed_id=seed_id)
            .delete()
        )
        logger.info(f"Deleted {deleted_count} existing endpoints for seed_id {seed_id}")

        # Add new endpoints
        now = datetime.now(timezone.utc)

        def add_endpoints(points, status):
            for point in points:
                endpoint = EndpointModel(
                    project_name=project_name,
                    seed_id=seed_id,
                    x=point[0],
                    y=point[1],
                    z=point[2],
                    status=status,
                    user=user_id,
                    created_at=now,
                    updated_at=now,
                )
                session.add(endpoint)

        # Add all endpoints
        add_endpoints(annotations["certain_ends"], "CERTAIN")
        add_endpoints(annotations["uncertain_ends"], "UNCERTAIN")
        add_endpoints(annotations["breadcrumbs"], "BREADCRUMB")

        session.commit()

        logger.info(f"Updated segment {seed_id} with status '{completion_status}'")

        # Auto-create postprocess task for completed traces
        if completion_status == "Done":
            try:
                # Generate unique task ID
                postprocess_task_id = f"postprocess_{seed_id}_{generate_id_nonunique()}"

                # Create minimal task data for postprocessing
                postprocess_task = Task(
                    task_id=postprocess_task_id,
                    task_type="trace_postprocess_v0",
                    ng_state={},  # Empty state, worker doesn't need it
                    ng_state_initial={},
                    completion_status="",
                    assigned_user_id="",
                    active_user_id="",
                    completed_user_id="",
                    priority=10,  # Low priority
                    batch_id="postprocess",
                    last_leased_ts=0.0,
                    is_active=True,
                    is_paused=False,
                    is_checked=False,
                    extra_data={
                        "original_task_id": task["task_id"],
                    },
                )

                # Create the task
                created_task_id = create_task(
                    project_name=project_name, data=postprocess_task, db_session=session
                )

                logger.info(f"Created postprocess task {created_task_id} for segment {seed_id}")

            except Exception as e:  # pylint: disable=broad-exception-caught
                # Don't fail the trace completion if postprocess creation fails
                logger.error(f"Failed to create postprocess task for segment {seed_id}: {e}")


@register_verifier("trace_v0")
def verify_trace_v0(  # pylint: disable=too-many-return-statements,too-many-branches
    project_name: str, task: Task, completion_status: str  # pylint: disable=unused-argument
) -> VerificationResult:
    """Verify trace_v0 task completion."""
    # For faulty tasks, skip all validation
    if completion_status == "Faulty Task":
        return VerificationResult(
            passed=True, message="Task marked as faulty - skipping validation"
        )

    ng_state = task.get("ng_state", {})

    # Use shared verification logic
    layers_result = verify_trace_layers(ng_state, require_single_segment=True)
    if not layers_result.passed:
        return layers_result

    # Status-specific validation
    if completion_status == "Done":
        # Check for Root Location layer
        root_location_layer = get_layer_by_name(ng_state, "Root Location")
        if not root_location_layer:
            return VerificationResult(
                passed=False, message="Status 'done' requires Root Location layer"
            )

        # Check exactly one point in Root Location
        root_annotations = count_annotations(root_location_layer)
        if root_annotations != 1:
            return VerificationResult(
                passed=False,
                message=f"Root Location must have exactly 1 annotation, found {root_annotations}",
            )

        # Check at least 10 certain ends
        certain_ends_layer = get_layer_by_name(ng_state, "Certain Ends")
        certain_ends_count = count_annotations(certain_ends_layer) if certain_ends_layer else 0
        if certain_ends_count < 2:
            return VerificationResult(
                passed=False, message="Insufficient certain end count for a complete neuron"
            )

        return VerificationResult(
            passed=True, message=f"Task completed with {certain_ends_count} certain ends"
        )

    elif completion_status == "Can't Continue":
        # Check at least 1 uncertain end
        uncertain_ends_layer = get_layer_by_name(ng_state, "Uncertain Ends")
        uncertain_ends_count = (
            count_annotations(uncertain_ends_layer) if uncertain_ends_layer else 0
        )
        if uncertain_ends_count < 1:
            return VerificationResult(
                passed=False,
                message=(
                    f"Status 'cant_continue' requires at least 1 uncertain end, "
                    f"found {uncertain_ends_count}"
                ),
            )

        return VerificationResult(
            passed=True,
            message=f"Task marked as cant_continue with {uncertain_ends_count} uncertain ends",
        )

    elif completion_status in ["Merger", "Wrong Cell Type"]:
        # These statuses just need the basic requirements (already checked above)
        return VerificationResult(passed=True, message=f"Task marked as {completion_status}")

    else:
        return VerificationResult(
            passed=False, message=f"Invalid completion status for trace_v0: {completion_status}"
        )


def generate_trace_v0_ng_state(project_name: str, segment: SegmentModel) -> dict:
    """Generate neuroglancer state for a trace_v0 task."""
    with get_session_context() as session:
        # Get project info
        project = session.query(ProjectModel).filter_by(project_name=project_name).first()
        if not project:
            raise ValueError(f"Project {project_name} not found")

        # Get existing endpoints for this segment
        endpoints = (
            session.query(EndpointModel)
            .filter_by(project_name=project_name, seed_id=segment.seed_id)
            .all()
        )

        # Group endpoints by status
        certain_ends = []
        uncertain_ends = []
        breadcrumbs = []

        for endpoint in endpoints:
            point = [endpoint.x, endpoint.y, endpoint.z]
            annotation = {
                "point": point,
                "type": "point",
                "id": str(uuid.uuid4()).replace("-", ""),
            }

            if endpoint.status == "CERTAIN":
                certain_ends.append(annotation)
            elif endpoint.status == "UNCERTAIN":
                uncertain_ends.append(annotation)
            elif endpoint.status == "BREADCRUMB":
                breadcrumbs.append(annotation)

        # Build neuroglancer state
        sv_resolution = [project.sv_resolution_x, project.sv_resolution_y, project.sv_resolution_z]

        ng_state: dict[str, Any] = {
            "dimensions": {
                "x": [sv_resolution[0] * 1e-9, "m"],
                "y": [sv_resolution[1] * 1e-9, "m"],
                "z": [sv_resolution[2] * 1e-9, "m"],
            },
            "position": [segment.seed_x, segment.seed_y, segment.seed_z],
            "crossSectionScale": 0.5,
            "projectionOrientation": [0.5, -0.5, 0.5, 0.5],
            "projectionScale": 2740.449487163767,
            "projectionDepth": 541651.3969244945,
            "layers": [
                {
                    "type": "image",
                    "source": (
                        "precomputed://gs://dkronauer-ant-001-alignment-asia-east2"
                        "/aligned/img_jpeg"
                    ),
                    "tab": "source",
                    "name": "Image",
                },
                {
                    "type": "segmentation",
                    "source": {
                        "url": project.segmentation_path
                        or (
                            "graphene://middleauth+https://data.proofreading.zetta.ai"
                            "/segmentation/table/kronauer_ant_x1"
                        ),
                        "state": {
                            "multicut": {"sinks": [], "sources": []},
                            "merge": {"merges": []},
                            "findPath": {},
                        },
                    },
                    "tab": "segments",
                    "segments": (
                        [str(segment.current_segment_id)] if segment.current_segment_id else []
                    ),
                    "name": "Segmentation",
                },
                {
                    "type": "annotation",
                    "source": {
                        "url": "local://annotations",
                        "transform": {
                            "outputDimensions": {
                                "x": [sv_resolution[0] * 1e-9, "m"],
                                "y": [sv_resolution[1] * 1e-9, "m"],
                                "z": [sv_resolution[2] * 1e-9, "m"],
                            }
                        },
                    },
                    "tool": "annotatePoint",
                    "tab": "annotations",
                    "annotationColor": "#ff00ff",  # Purple
                    "annotations": [
                        {
                            "point": [segment.seed_x, segment.seed_y, segment.seed_z],
                            "type": "point",
                            "id": str(uuid.uuid4()).replace("-", ""),
                        }
                    ],
                    "name": "Seed Location",
                },
                {
                    "type": "annotation",
                    "source": {
                        "url": "local://annotations",
                        "transform": {
                            "outputDimensions": {
                                "x": [sv_resolution[0] * 1e-9, "m"],
                                "y": [sv_resolution[1] * 1e-9, "m"],
                                "z": [sv_resolution[2] * 1e-9, "m"],
                            }
                        },
                    },
                    "tool": "annotatePoint",
                    "tab": "annotations",
                    "annotationColor": "#00ff00",  # Green
                    "annotations": [],
                    "name": "Root Location",
                },
                {
                    "type": "annotation",
                    "source": {
                        "url": "local://annotations",
                        "transform": {
                            "outputDimensions": {
                                "x": [sv_resolution[0] * 1e-9, "m"],
                                "y": [sv_resolution[1] * 1e-9, "m"],
                                "z": [sv_resolution[2] * 1e-9, "m"],
                            }
                        },
                    },
                    "tool": "annotatePoint",
                    "tab": "annotations",
                    "annotationColor": "#ffff00",  # Yellow
                    "annotations": certain_ends,
                    "name": "Certain Ends",
                },
                {
                    "type": "annotation",
                    "source": {
                        "url": "local://annotations",
                        "transform": {
                            "outputDimensions": {
                                "x": [sv_resolution[0] * 1e-9, "m"],
                                "y": [sv_resolution[1] * 1e-9, "m"],
                                "z": [sv_resolution[2] * 1e-9, "m"],
                            }
                        },
                    },
                    "tool": "annotatePoint",
                    "tab": "annotations",
                    "annotationColor": "#ff0000",  # Red
                    "annotations": uncertain_ends,
                    "name": "Uncertain Ends",
                },
                {
                    "type": "annotation",
                    "source": {
                        "url": "local://annotations",
                        "transform": {
                            "outputDimensions": {
                                "x": [sv_resolution[0] * 1e-9, "m"],
                                "y": [sv_resolution[1] * 1e-9, "m"],
                                "z": [sv_resolution[2] * 1e-9, "m"],
                            }
                        },
                    },
                    "tool": "annotatePoint",
                    "tab": "rendering",
                    "annotationColor": "#0400ff",  # Blue
                    "annotations": breadcrumbs,
                    "name": "Breadcrumbs",
                },
            ],
            "showSlices": False,
            "selectedLayer": {"visible": True, "layer": "Segmentation"},
            "layout": "xy-3d",
        }

        # Add extra layers if configured in project
        if project.extra_layers:
            for layer in project.extra_layers:
                ng_state["layers"].append(layer)

        return ng_state


@register_creation_handler("trace_v0")
def create_trace_v0_task(project_name: str, segment: SegmentModel, kwargs: dict) -> str:
    """Create a trace_v0 task for a segment.

    Args:
        project_name: The project name
        segment: The segment to create a task for
        kwargs: Optional kwargs including priority, batch_id, etc.

    Returns:
        The created task_id
    """
    # Generate unique task ID
    task_id = f"trace_{segment.seed_id}_{generate_id_nonunique()}"

    # Generate neuroglancer state
    ng_state = generate_trace_v0_ng_state(project_name, segment)

    # Build task data
    task_data = Task(
        task_id=task_id,
        task_type="trace_v0",
        ng_state=ng_state,
        ng_state_initial=copy.deepcopy(ng_state),
        completion_status="",
        assigned_user_id="",
        active_user_id="",
        completed_user_id="",
        priority=kwargs.get("priority", 50),
        batch_id=kwargs.get("batch_id", "default"),
        last_leased_ts=0.0,
        is_active=True,
        is_paused=kwargs.get("is_paused", False),
        is_checked=False,
        extra_data={"seed_id": segment.seed_id},
    )

    # Create the task
    with get_session_context() as session:
        created_task_id = create_task(
            project_name=project_name, data=task_data, db_session=session
        )

        # Update segment's task_ids
        seg = (
            session.query(SegmentModel)
            .filter_by(project_name=project_name, seed_id=segment.seed_id)
            .first()
        )

        if seg and task_id not in seg.task_ids:
            seg.task_ids = seg.task_ids + [task_id]
            seg.updated_at = datetime.now(timezone.utc)
            session.commit()
            logger.info(f"Linked task {task_id} to segment {segment.seed_id}")

    return created_task_id
