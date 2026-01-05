"""
Ingest segment coordinates from file into database with two-phase processing.

This script implements a two-phase approach:
1. Validation phase: validates all coordinates and checks for conflicts
2. Ingestion phase: creates segments only after all validations pass

Uses individual segment ID lookups for memory efficiency and simplicity.
"""

import json
import os
import sys
from typing import Any

import click

from zetta_utils import log
from zetta_utils.task_management.db.models import SegmentModel, SegmentTypeModel
from zetta_utils.task_management.db.session import get_session_context
from zetta_utils.task_management.segment import (
    create_segment_from_coordinate,
    get_segment_id,
)

logger = log.get_logger()


def load_coordinates_from_file(coordinates_file: str) -> list[list[float]]:
    """
    Load coordinates from JSON file, handling multiple formats.

    Args:
        coordinates_file: Path to JSON file containing coordinates

    Returns:
        List of [x, y, z] coordinates
    """
    try:
        with open(coordinates_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        coordinates = []

        if isinstance(data, dict) and "segments" in data:
            # Format from get_seg_type_points output
            for seg_id, seg_data in data["segments"].items():
                if isinstance(seg_data, dict) and "centroid" in seg_data:
                    coordinates.append(seg_data["centroid"])
                elif isinstance(seg_data, list):
                    coordinates.append(seg_data)
                else:
                    logger.warning(f"Skipping invalid segment data for {seg_id}: {seg_data}")
        elif isinstance(data, list):
            # Simple list of coordinates
            coordinates = data
        else:
            raise ValueError(
                "Invalid file format. Expected list of coordinates or dict with 'segments' key"
            )

        # Validate and convert coordinates
        validated_coords = []
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, list) or len(coord) != 3:
                raise ValueError(f"Invalid coordinate format at index {i}: {coord}")
            try:
                validated_coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid coordinate values at index {i}: {coord}") from exc

        logger.info(f"Loaded {len(validated_coords)} coordinates from {coordinates_file}")
        return validated_coords

    except Exception as e:
        raise ValueError(f"Failed to load coordinates from {coordinates_file}: {e}") from e


def get_segment_ids_for_coordinates(
    project_name: str, coordinates: list[list[float]], db_session: Any = None
) -> dict[tuple[float, float, float], int]:
    """
    Get segment IDs for all coordinates using individual lookups.

    Args:
        project_name: Name of the project
        coordinates: List of [x, y, z] coordinates
        db_session: Optional database session

    Returns:
        Dictionary mapping coordinates to segment IDs
    """
    result: dict[tuple[float, float, float], int] = {}

    logger.info(f"Looking up segment IDs for {len(coordinates)} coordinates...")

    for i, coord in enumerate(coordinates):
        coord_tuple: tuple[float, float, float] = (coord[0], coord[1], coord[2])

        segment_id = get_segment_id(
            project_name=project_name, coordinate=coord, initial=False, db_session=db_session
        )
        result[coord_tuple] = segment_id

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(coordinates)} coordinates")

    logger.info(f"Retrieved segment IDs for {len(coordinates)} coordinates")
    return result


def validate_coordinates_batch(
    project_name: str,
    coordinates: list[list[float]],
    expected_neuron_type: str,
    segment_ids: dict[tuple[float, float, float], int],
    db_session: Any = None,
) -> dict[str, Any]:
    """
    Validate all coordinates and check for conflicts.

    Args:
        project_name: Name of the project
        coordinates: List of coordinates to validate
        expected_neuron_type: Expected neuron type for all segments
        segment_ids: Mapping of coordinates to segment IDs
        db_session: Optional database session

    Returns:
        Validation results dictionary
    """
    logger.info("Starting validation phase...")

    # Validate neuron type exists
    with get_session_context(db_session) as session:
        neuron_type = (
            session.query(SegmentTypeModel)
            .filter(
                SegmentTypeModel.project_name == project_name,
                SegmentTypeModel.type_name == expected_neuron_type,
            )
            .first()
        )

        if not neuron_type:
            raise ValueError(
                f"Expected neuron type '{expected_neuron_type}' not found in "
                f"project '{project_name}'"
            )

    # Prepare validation results
    validation_results: dict[str, Any] = {
        "total_coordinates": len(coordinates),
        "valid_coordinates": [],
        "invalid_coordinates": [],
        "existing_segments": [],
        "type_conflicts": [],
        "errors": [],
    }

    # Get all unique segment IDs that are non-zero
    valid_segment_ids = {seg_id for seg_id in segment_ids.values() if seg_id != 0}

    if valid_segment_ids:
        logger.info(f"Checking {len(valid_segment_ids)} unique segment IDs in database...")

        # Query existing segments in batch
        with get_session_context(db_session) as session:
            existing_segments = (
                session.query(SegmentModel)
                .filter(
                    SegmentModel.project_name == project_name,
                    SegmentModel.current_segment_id.in_(list(valid_segment_ids)),
                )
                .all()
            )

            # Create mapping of segment_id to existing segment
            existing_segments_map = {seg.current_segment_id: seg for seg in existing_segments}
            logger.info(f"Found {len(existing_segments_map)} existing segments in database")
    else:
        existing_segments_map = {}

    # Validate each coordinate
    for coord in coordinates:
        coord_tuple: tuple[float, float, float] = (coord[0], coord[1], coord[2])
        segment_id = segment_ids.get(coord_tuple, 0)

        if segment_id == 0:
            validation_results["invalid_coordinates"].append(
                {"coordinate": coord, "error": "No segment found at coordinate"}
            )
            validation_results["errors"].append(f"No segment found at coordinate {coord}")
            continue

        # Check if segment exists in database
        existing_segment = existing_segments_map.get(segment_id)

        if existing_segment:
            # Check for type conflict
            if (
                existing_segment.expected_segment_type
                and existing_segment.expected_segment_type != expected_neuron_type
            ):

                conflict_info = {
                    "coordinate": coord,
                    "segment_id": segment_id,
                    "existing_type": existing_segment.expected_segment_type,
                    "requested_type": expected_neuron_type,
                }
                validation_results["type_conflicts"].append(conflict_info)
                validation_results["errors"].append(
                    f"Type conflict for segment {segment_id}: "
                    f"existing={existing_segment.expected_segment_type}, "
                    f"requested={expected_neuron_type}"
                )
            else:
                validation_results["existing_segments"].append(
                    {"coordinate": coord, "segment_id": segment_id, "status": "matching_type"}
                )
        else:
            # New segment to be created
            validation_results["valid_coordinates"].append(
                {"coordinate": coord, "segment_id": segment_id}
            )

    # Log validation summary
    logger.info("Validation complete:")
    logger.info(f"  Total coordinates: {validation_results['total_coordinates']}")
    logger.info(f"  Valid for creation: {len(validation_results['valid_coordinates'])}")
    logger.info(f"  Existing segments: {len(validation_results['existing_segments'])}")
    logger.info(f"  Type conflicts: {len(validation_results['type_conflicts'])}")
    logger.info(f"  Invalid coordinates: {len(validation_results['invalid_coordinates'])}")

    return validation_results


def ingest_validated_coordinates(
    project_name: str,
    valid_coordinates: list[dict],
    expected_neuron_type: str,
    batch_name: str,
    db_session: Any = None,
) -> dict[str, Any]:
    """
    Ingest coordinates that have been validated.

    Args:
        project_name: Name of the project
        valid_coordinates: List of validated coordinate dictionaries
        expected_neuron_type: Expected neuron type for all segments
        batch_name: Batch identifier for created segments
        db_session: Optional database session

    Returns:
        Ingestion results dictionary
    """
    logger.info("Starting ingestion phase...")

    results: dict[str, Any] = {
        "created_segments": 0,
        "creation_errors": 0,
        "created_seed_ids": [],
        "errors": [],
    }

    for coord_data in valid_coordinates:
        coordinate = coord_data["coordinate"]

        segment = create_segment_from_coordinate(
            project_name=project_name,
            coordinate=coordinate,
            batch=batch_name,
            expected_segment_type=expected_neuron_type,
            db_session=db_session,
        )

        results["created_segments"] += 1
        results["created_seed_ids"].append(segment["seed_id"])
        logger.debug(f"Created segment with seed_id {segment['seed_id']} at {coordinate}")

        # Progress logging
        if results["created_segments"] % 100 == 0:
            logger.info(f"Created {results['created_segments']}/{len(valid_coordinates)} segments")

    logger.info(f"Ingestion complete: created {results['created_segments']} segments")
    return results


@click.command()
@click.argument("project_name")
@click.argument("coordinates_file", type=click.Path(exists=True))
@click.argument("expected_neuron_type")
@click.argument("batch_name")
@click.option("--output", "-o", help="Path to save results JSON")
@click.option(
    "--fail-on-conflicts/--no-fail-on-conflicts",
    default=True,
    help="Whether to fail on type conflicts (default: fail)",
)
def main(
    project_name: str,
    coordinates_file: str,
    expected_neuron_type: str,
    batch_name: str,
    output: str | None,
    fail_on_conflicts: bool,
):
    """
    Ingest segment coordinates from file into database with two-phase processing.

    Phase 1: Validates all coordinates and checks for conflicts
    Phase 2: Creates segments only after all validations pass

    Uses individual segment ID lookups for memory efficiency.
    """
    try:
        print(f"Processing coordinates from: {coordinates_file}")

        # Load coordinates
        coordinates = load_coordinates_from_file(coordinates_file)

        # Get segment IDs for all coordinates
        segment_ids = get_segment_ids_for_coordinates(project_name, coordinates)

        # Phase 1: Validation
        print("Phase 1: Validating coordinates...")
        validation_results = validate_coordinates_batch(
            project_name=project_name,
            coordinates=coordinates,
            expected_neuron_type=expected_neuron_type,
            segment_ids=segment_ids,
        )

        # Check for conflicts
        if fail_on_conflicts and validation_results["type_conflicts"]:
            print(f"\nERROR: Found {len(validation_results['type_conflicts'])} type conflicts!")
            for conflict in validation_results["type_conflicts"][:5]:
                print(
                    f"  Segment {conflict['segment_id']}: "
                    f"{conflict['existing_type']} vs {conflict['requested_type']}"
                )
            if len(validation_results["type_conflicts"]) > 5:
                print(
                    f"  ... and {len(validation_results['type_conflicts']) - 5} " f"more conflicts"
                )
            sys.exit(1)

        # Check for other validation errors
        if validation_results["invalid_coordinates"]:
            print(
                f"\nWARNING: Found {len(validation_results['invalid_coordinates'])} "
                f"invalid coordinates"
            )

        # Phase 2: Ingestion
        if validation_results["valid_coordinates"]:
            print(f"Phase 2: Creating {len(validation_results['valid_coordinates'])} segments...")
            ingestion_results = ingest_validated_coordinates(
                project_name=project_name,
                valid_coordinates=validation_results["valid_coordinates"],
                expected_neuron_type=expected_neuron_type,
                batch_name=batch_name,
            )
        else:
            print("No valid coordinates to ingest.")
            ingestion_results = {
                "created_segments": 0,
                "creation_errors": 0,
                "created_seed_ids": [],
                "errors": [],
            }

        # Combine results
        final_results = {
            "project_name": project_name,
            "expected_neuron_type": expected_neuron_type,
            "batch_name": batch_name,
            "coordinates_file": coordinates_file,
            **validation_results,
            **ingestion_results,
        }

        # Save results
        if output:
            abs_path = os.path.abspath(output)
            print(f"Writing results to file: {abs_path}")
            with open(output, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2)
            print(f"Results saved to: {abs_path}")

        # Print summary
        print(f"\nIngestion Summary for {project_name}:")
        print(f"  Expected type: {expected_neuron_type}")
        print(f"  Batch: {batch_name}")
        print(f"  Total coordinates: {final_results['total_coordinates']}")
        print(f"  Created segments: {final_results['created_segments']}")
        print(f"  Existing segments: {len(final_results['existing_segments'])}")
        print(f"  Type conflicts: {len(final_results['type_conflicts'])}")
        print(f"  Invalid coordinates: {len(final_results['invalid_coordinates'])}")
        print(f"  Creation errors: {final_results['creation_errors']}")

        if final_results["created_segments"] > 0:
            print(f"\nSuccessfully created {final_results['created_segments']} new segments!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise click.ClickException(str(e))
