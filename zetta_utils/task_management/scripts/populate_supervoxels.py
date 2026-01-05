#!/usr/bin/env python3
"""
Script to populate the supervoxels table from existing segments.

This script reads all segments from the segments table and creates
corresponding entries in the supervoxels table. It uses seed_id as
the supervoxel_id and current_segment_id from the segment.

Usage:
    python -m zetta_utils.task_management.scripts.populate_supervoxels \
        [--project-name PROJECT_NAME] \
        [--batch-size 1000] \
        [--dry-run]
"""

import argparse
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from zetta_utils.task_management.db.models import SegmentModel, SupervoxelModel
from zetta_utils.task_management.db.session import get_session_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def populate_supervoxels_from_segments(
    project_name: str | None = None,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Populate supervoxels table from existing segments.

    Args:
        project_name: Optional project name to filter by
        batch_size: Number of records to process per batch
        dry_run: If True, only count records without inserting

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_segments": 0,
        "segments_with_seed_id": 0,
        "segments_with_current_segment_id": 0,
        "supervoxels_created": 0,
        "supervoxels_skipped": 0,
        "errors": 0,
    }

    with get_session_context() as session:
        # Build query
        query = select(SegmentModel)
        if project_name:
            query = query.where(SegmentModel.project_name == project_name)

        # Get total count
        result = session.execute(query)
        segments = list(result.scalars().all())
        stats["total_segments"] = len(segments)

        logger.info(f"Found {stats['total_segments']} segments to process")

        # Process in batches
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1} "
                f"({i + 1}-{min(i + batch_size, len(segments))} "
                f"of {len(segments)})"
            )

            for segment in batch:
                # Check if segment has required fields
                if segment.seed_id is None:
                    stats["supervoxels_skipped"] += 1
                    logger.debug(
                        f"Skipping segment {segment.project_name} "
                        f"- no seed_id"
                    )
                    continue

                stats["segments_with_seed_id"] += 1

                if segment.current_segment_id is None:
                    stats["supervoxels_skipped"] += 1
                    logger.debug(
                        f"Skipping segment {segment.project_name}/{segment.seed_id} "
                        f"- no current_segment_id"
                    )
                    continue

                stats["segments_with_current_segment_id"] += 1

                if dry_run:
                    logger.debug(
                        f"Would create supervoxel: {segment.seed_id} -> "
                        f"{segment.current_segment_id}"
                    )
                    stats["supervoxels_created"] += 1
                    continue

                # Check if supervoxel already exists
                existing = session.execute(
                    select(SupervoxelModel).where(
                        SupervoxelModel.supervoxel_id == segment.seed_id
                    )
                ).scalar_one_or_none()

                if existing:
                    logger.debug(
                        f"Supervoxel {segment.seed_id} already exists, skipping"
                    )
                    stats["supervoxels_skipped"] += 1
                    continue

                # Create supervoxel
                try:
                    now = datetime.now(timezone.utc)
                    supervoxel = SupervoxelModel(
                        supervoxel_id=segment.seed_id,
                        seed_x=segment.seed_x,
                        seed_y=segment.seed_y,
                        seed_z=segment.seed_z,
                        current_segment_id=segment.current_segment_id,
                        created_at=segment.created_at or now,
                        updated_at=segment.updated_at or now,
                    )
                    session.add(supervoxel)
                    stats["supervoxels_created"] += 1

                    logger.debug(
                        f"Created supervoxel {segment.seed_id} -> "
                        f"{segment.current_segment_id}"
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    stats["errors"] += 1
                    logger.error(
                        f"Error creating supervoxel for segment "
                        f"{segment.seed_id}: {e}"
                    )

            # Commit batch
            if not dry_run:
                try:
                    session.commit()
                    logger.info(f"Committed batch of {len(batch)} segments")
                except Exception as e:  # pylint: disable=broad-exception-caught
                    stats["errors"] += 1
                    logger.error(f"Error committing batch: {e}")
                    session.rollback()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Populate supervoxels table from existing segments"
    )
    parser.add_argument(
        "--project-name",
        help="Filter by project name (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - count records without inserting",
    )

    args = parser.parse_args()

    logger.info("Starting supervoxel population")
    logger.info(f"Project: {args.project_name or 'ALL'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dry run: {args.dry_run}")

    stats = populate_supervoxels_from_segments(
        project_name=args.project_name,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("SUMMARY:")
    logger.info(f"Total segments: {stats['total_segments']}")
    logger.info(f"Segments with seed_id: {stats['segments_with_seed_id']}")
    logger.info(
        f"Segments with current_segment_id: {stats['segments_with_current_segment_id']}"
    )
    logger.info(f"Supervoxels created: {stats['supervoxels_created']}")
    logger.info(f"Supervoxels skipped: {stats['supervoxels_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("DRY RUN - No changes were made to the database")
    else:
        logger.info("Supervoxel population complete!")


if __name__ == "__main__":
    main()
