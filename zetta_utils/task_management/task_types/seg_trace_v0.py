import copy
from datetime import datetime

from ..types import Task
from .completion import register_completion_handler, CompletionResult
from .verification import register_verifier, VerificationResult


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


@register_completion_handler("seg_trace_v0")
def handle_seg_trace_v0_completion(task: Task, completion_status: str) -> CompletionResult:
    """Handle completion of seg_trace_v0 tasks.
    
    For 'done' status:
    - Extract segment ID and root location
    - Update any downstream tasks or data
    
    For 'cant_continue', 'merger', 'wrong_cell_type':
    - Mark segment as incomplete/problematic
    """
    ng_state = task.get("ng_state", {})
    updated_ng_state = copy.deepcopy(ng_state)
    
    # Get the visible segment ID
    segmentation_layer = get_layer_by_name(ng_state, "Segmentation")
    if not segmentation_layer:
        return CompletionResult(
            success=False,
            message="No Segmentation layer found in ng_state"
        )
    
    segment_id = get_visible_segment_id(segmentation_layer)
    if not segment_id:
        return CompletionResult(
            success=False,
            message="No visible segment found in Segmentation layer"
        )
    
    # Add completion metadata to ng_state
    if "metadata" not in updated_ng_state:
        updated_ng_state["metadata"] = {}
    
    updated_ng_state["metadata"]["completion_status"] = completion_status
    updated_ng_state["metadata"]["completed_at"] = datetime.utcnow().isoformat()
    updated_ng_state["metadata"]["segment_id"] = segment_id
    
    if completion_status == "done":
        # Extract root location for downstream use
        root_location_layer = get_layer_by_name(ng_state, "Root Location")
        if root_location_layer and root_location_layer.get("annotations"):
            root_annotation = root_location_layer["annotations"][0]
            root_point = root_annotation.get("point", [])
            updated_ng_state["metadata"]["root_location"] = root_point
            
            # Extract certain ends count
            certain_ends_layer = get_layer_by_name(ng_state, "Certain Ends")
            certain_ends_count = len(certain_ends_layer.get("annotations", []))
            updated_ng_state["metadata"]["certain_ends_count"] = certain_ends_count
            
            message = f"Completed segment {segment_id} with {certain_ends_count} certain ends"
        else:
            message = f"Completed segment {segment_id} (no root location found)"
            
    elif completion_status == "cant_continue":
        # Extract uncertain ends for analysis
        uncertain_ends_layer = get_layer_by_name(ng_state, "Uncertain Ends")
        uncertain_ends_count = len(uncertain_ends_layer.get("annotations", []))
        updated_ng_state["metadata"]["uncertain_ends_count"] = uncertain_ends_count
        message = f"Segment {segment_id} marked as cant_continue with {uncertain_ends_count} uncertain ends"
        
    elif completion_status in ["merger", "wrong_cell_type"]:
        message = f"Segment {segment_id} marked as {completion_status}"
        
    else:
        return CompletionResult(
            success=False,
            message=f"Invalid completion status: {completion_status}"
        )
    
    return CompletionResult(
        success=True,
        message=message,
        updated_ng_state=updated_ng_state
    )


@register_verifier("seg_trace_v0")
def verify_seg_trace_v0(task: Task, completion_status: str) -> VerificationResult:
    """Verify seg_trace_v0 task completion."""
    ng_state = task.get("ng_state", {})
    
    # Check required layers exist
    required_layers = ["Segmentation", "Breadcrumbs", "Certain Ends", "Uncertain Ends", "Seed Location"]
    missing_layers = []
    
    for layer_name in required_layers:
        if not get_layer_by_name(ng_state, layer_name):
            missing_layers.append(layer_name)
    
    if missing_layers:
        return VerificationResult(
            passed=False,
            message=f"Missing required layers: {', '.join(missing_layers)}"
        )
    
    # Check exactly one visible segment in Segmentation layer
    segmentation_layer = get_layer_by_name(ng_state, "Segmentation")
    visible_segments = count_visible_segments(segmentation_layer)
    
    if visible_segments != 1:
        return VerificationResult(
            passed=False,
            message=f"Segmentation layer must have exactly 1 visible segment, found {visible_segments}"
        )
    
    # Status-specific validation
    if completion_status == "done":
        # Check for Root Location layer
        root_location_layer = get_layer_by_name(ng_state, "Root Location")
        if not root_location_layer:
            return VerificationResult(
                passed=False,
                message="Status 'done' requires Root Location layer"
            )
        
        # Check exactly one point in Root Location
        root_annotations = count_annotations(root_location_layer)
        if root_annotations != 1:
            return VerificationResult(
                passed=False,
                message=f"Root Location must have exactly 1 annotation, found {root_annotations}"
            )
        
        # Check at least 10 certain ends
        certain_ends_layer = get_layer_by_name(ng_state, "Certain Ends")
        certain_ends_count = count_annotations(certain_ends_layer)
        if certain_ends_count < 10:
            return VerificationResult(
                passed=False,
                message=f"Insufficient certain end count for a complete neuron"
            )
        
        return VerificationResult(
            passed=True,
            message=f"Task completed with {certain_ends_count} certain ends"
        )
    
    elif completion_status == "cant_continue":
        # Check at least 1 uncertain end
        uncertain_ends_layer = get_layer_by_name(ng_state, "Uncertain Ends")
        uncertain_ends_count = count_annotations(uncertain_ends_layer)
        if uncertain_ends_count < 1:
            return VerificationResult(
                passed=False,
                message=f"Status 'cant_continue' requires at least 1 uncertain end, found {uncertain_ends_count}"
            )
        
        return VerificationResult(
            passed=True,
            message=f"Task marked as cant_continue with {uncertain_ends_count} uncertain ends"
        )
    
    elif completion_status in ["merger", "wrong_cell_type"]:
        # These statuses just need the basic requirements (already checked above)
        return VerificationResult(
            passed=True,
            message=f"Task marked as {completion_status}"
        )
    
    else:
        return VerificationResult(
            passed=False,
            message=f"Invalid completion status for seg_trace_v0: {completion_status}"
        )