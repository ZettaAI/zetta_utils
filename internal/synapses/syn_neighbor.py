"""
This module provides a VolumetricOp does synapse assignment via a nearest-neighbor algorithm:

1. Caller should use generous padding on the processing chunks.
2. Calculate the centroid of each cluster; if it’s in the padding area, skip it
   (it will be handled by the neighboring chunk).
3. Find the Z layer with maximum area in the cluster; remaining work is done in 2D at this Z.
4. Find the ID of the cell most overlapping the cluster; this is one end of the synapse.
5. Dilate the cluster (in 2D) by 5 pixels or so; using the cell segmentation, see what cell
   (other than the one found above) most overlaps now; this is the other end of the synapse.
6. If step 5 doesn’t overlap a new cell ID, then repeat (dilate and check) until it does.
"""
import math
from typing import List, Optional, Sequence, Tuple, Union

import attrs
import numpy as np
from scipy import ndimage

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.internal.synapses.syn_assignment import (
    bfs_nearest_value,
    centroid_of_id,
)
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.layer.volumetric.annotation.backend import (
    AnnotationLayerBackend,
    LineAnnotation,
)

logger = log.get_logger("zetta_utils")


def magnitude(vec):
    return math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)


def normalized(vec):
    return vec / magnitude(vec)


def most_overlapping_id(id_array, mask, excluding_id=None, valid_ids=None):
    """
    Return the nonzero value in id_array which most commonly occurs in the
    places corresponding to mask is True.

    If excluding_id is given, ignore that value (as well as 0).

    If valid_ids is given, also ignore any values not in this list.

    If there are acceptable values within the mask, return None.
    """
    overlapping_ids = id_array[mask]
    overlapping_ids = overlapping_ids[overlapping_ids != 0]  # Remove zeros
    if excluding_id is not None:
        overlapping_ids = overlapping_ids[overlapping_ids != excluding_id]
    if valid_ids is not None:
        overlapping_ids = overlapping_ids[np.isin(overlapping_ids, valid_ids)]

    if len(overlapping_ids) > 0:
        unique_vals, counts = np.unique(overlapping_ids, return_counts=True)
        return int(unique_vals[np.argmax(counts)])
    else:
        return None


def fast_dilate(
    mask: np.ndarray,
    distance_in_nm: float,
    resolution: Union[Tuple[float, float, float], List[float], Vec3D],
):
    """
    Fast 3D dilation using a Euclidean distance transform convolution.

    :param mask: 3D numpy array to dilate.
    :param distance_in_nm: Distance in nanometers to dilate.
    :param resolution: Voxel size in nanometers, specified as (x, y, z).
    :return: Dilated 3D numpy array.
    """
    # Compute distance transform of synapse mask
    dist_transform = ndimage.distance_transform_edt(1 - mask, sampling=resolution)

    # Create mask of voxels within distance
    within_distance = dist_transform <= distance_in_nm

    return within_distance.astype(int)


@builder.register("AssignNearestNeighborOp")
@mazepa.taskable_operation_cls
@attrs.frozen()
class AssignNearestNeighborOp:  # implements VolumetricOpProtocol
    crop_pad: Sequence[int] = (0, 0, 0)

    # amount (in nm) by which to dilate synapse mask (in X and Y) on
    # each step, when looking for overlapping cell IDs
    dilation_step_nm: int = 50

    # if not None, this is a list of cell IDs which are acceptable synapse
    # partners; all other IDs will be ignored
    valid_partner_ids: Sequence[int] | None = None

    # pylint: disable=no-self-use
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        # For simplicity, just return the destination resolution
        return dst_resolution

    # pylint: disable=unused-argument
    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "AssignNearestNeighborOp":
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    # pylint: disable=redefined-outer-name
    def __call__(
        self,
        idx: VolumetricIndex,
        dst_lines: AnnotationLayerBackend,
        src_synseg: VolumetricLayer,
        src_cellseg: VolumetricLayer,
        *args,
        **kwargs,
    ):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        # logger.info(
        #     f"Op called with dst_lines: {dst_lines}, idx: {idx}, crop_pad: {self.crop_pad}"
        # )

        # Load data from our two input layers, into numpy arrays
        padded_idx = idx.padded(self.crop_pad)
        synseg_data = src_synseg[padded_idx][0]
        cellseg_data = src_cellseg[padded_idx][0]
        data_start = padded_idx.start
        shape: Vec3D = Vec3D(*synseg_data.shape)
        crop_pad_vec: Vec3D = Vec3D(*self.crop_pad)

        # Iterate over the synapse IDs
        syn_ids = np.unique(synseg_data)
        syn_ids = syn_ids[syn_ids != 0]
        logger.info(f"Chunk {idx.chunk_id} contains {len(syn_ids)} synapses")
        # pylint: disable=consider-using-with
        line_annotations = []
        for syn_id in syn_ids:
            # Find centroid; if it's in the padding area, skip it.
            centroid: Vec3D = Vec3D(*centroid_of_id(synseg_data, syn_id))
            in_padding_area = np.any((centroid < crop_pad_vec) | (centroid > shape - crop_pad_vec))
            if in_padding_area:
                continue

            # Find the Z-slice with the maximum amount of syn_id in it.
            best_z = np.argmax(np.sum(synseg_data == syn_id, axis=(0, 1)))
            synseg_2d = synseg_data[:, :, best_z]
            cellseg_2d = cellseg_data[:, :, best_z]

            # Find the cell that most overlaps our synapse area (mask).
            # This identifies the cell on this side of the synapse.
            synapse_mask = synseg_2d == syn_id
            source_cell_id = most_overlapping_id(cellseg_2d, synapse_mask)
            if source_cell_id is None:
                logger.warning(
                    f"Synapse {syn_id} overlaps no cell at {tuple(data_start[0] + centroid)}"
                )
                continue

            # Redefine the centroid using this 2D slice.  This will be our starting
            # point for finding the nearest neighboring cell.
            x_coords, y_coords = np.nonzero(synapse_mask)
            centroid = Vec3D(round(np.mean(x_coords)), round(np.mean(y_coords)), best_z)
            # print(f'Centroid there is {centroid} + {data_start} = {centroid + data_start}')

            # Now, dilate the mask, and check for the most-overlapping cell that
            # is not our source cell.  Repeat some reasonable number of times
            # until we get a valid result.  This identifies the cell on the other
            # side of the synapse.
            dest_cell_id = None
            max_iterations = 10
            for i in range(1, max_iterations + 1):
                synapse_mask = fast_dilate(synapse_mask, self.dilation_step_nm * i, idx.resolution)
                dest_cell_id = most_overlapping_id(
                    cellseg_2d, synapse_mask, source_cell_id, self.valid_partner_ids
                )
                if dest_cell_id is not None:
                    break
            if dest_cell_id is None:
                continue

            # Write to CSV
            # csv_file.write(f"{syn_id},{source_cell_id},{dest_cell_id}\n")

            # Also create an annotation for the precomputed lines file.
            # We'll use the centroid on one side, and the nearest point in the
            # neighboring cell for the other side (extended a few pixels to make
            # it easier to see, and more robust to cell segmentation tweaks.)
            source_point = bfs_nearest_value(cellseg_data, centroid, source_cell_id, True)
            if source_point is None:
                source_point = bfs_nearest_value(cellseg_data, centroid, source_cell_id, False)
            source_point = Vec3D(*source_point) + data_start

            dest_point = bfs_nearest_value(cellseg_data, centroid, dest_cell_id, True)
            if dest_point is None:
                dest_point = bfs_nearest_value(cellseg_data, centroid, dest_cell_id, False)
            dest_point = Vec3D(*dest_point) + data_start
            dest_point += normalized(dest_point - source_point) * 3
            line_annotations.append(LineAnnotation(syn_id, source_point, dest_point))

        # write precomputed annotations to smallest (lowest-level) chunks ONLY
        dst_lines.write_annotations(line_annotations, all_levels=False)


def nearest_value(
    array: np.ndarray, start: Sequence[int], target_value: int, xy_only: bool = False
):
    """
    Find the point in ``array`` with value ``target_value`` that is the closest
    (minimum Euclidean distance) to ``start``.

    :param array: N-dimensional numpy array to search in.
    :param start: Starting coordinates with the same dimensionality as the array.
    :param target_value: Value to search for.
    :param xy_only: If True, only consider points in the same Z plane as start.
    :return: (x, y, z) location of the point found, or None if no point is found.
    """
    # Create mask for target value
    mask = array == target_value
    if not np.any(mask):
        return None

    if xy_only:
        # Restrict search to the Z plane of start point
        z_mask = np.zeros_like(mask, dtype=bool)
        z_mask[:, :, int(start[2])] = True
        mask = mask & z_mask

    # Create coordinate grids
    grids = np.ogrid[tuple(slice(0, s) for s in array.shape)]

    # Calculate squared distances at each point
    dist_squared = sum((grid - coord) ** 2 for grid, coord in zip(grids, start))

    # Set distances to infinity where value doesn't match target
    distances = np.where(mask, dist_squared, np.inf)

    # Find minimum distance location
    min_idx = np.argmin(distances)
    # Convert flat index to coordinates
    return np.unravel_index(min_idx, array.shape)


@builder.register("AssignAllNeighborsOp")
@mazepa.taskable_operation_cls
@attrs.frozen()
class AssignAllNeighborsOp:  # implements VolumetricOpProtocol
    crop_pad: Sequence[int] = (0, 0, 0)

    # We assign synapses to all cells within this distance (in num) of the
    # edge of the input cluster (except for the cell containing that centroid).
    # (Note that padding of at least this size should also be applied.)
    max_distance: float = 500.0

    # How far (in XY pixels) to shift the endpoint of the line further beyond
    # the nearest point in the target cell, just to make it easier to see that
    # it really is within that cell, and not just sitting on the border.
    endpoint_shift: float = 4

    # If not None, this is a list of cell IDs which are acceptable cell
    # segment IDs for the source end of each synapse; any syn segments
    # that map to a cell not in this list will be ignored.
    valid_source_ids: Sequence[int] | None = None

    # pylint: disable=no-self-use
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        # For simplicity, just return the destination resolution
        return dst_resolution

    # pylint: disable=unused-argument
    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "AssignAllNeighborsOp":
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    # pylint: disable=redefined-outer-name
    def __call__(
        self,
        idx: VolumetricIndex,
        dst_lines: AnnotationLayerBackend,
        src_synseg: VolumetricLayer,
        src_cellseg: VolumetricLayer,
        *args,
        **kwargs,
    ):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        logger.info(
            f"Op called with dst_lines: {dst_lines}, idx: {idx}, crop_pad: {self.crop_pad}"
        )

        # Load data from our two input layers, into numpy arrays
        padded_idx = idx.padded(self.crop_pad)
        synseg_data = src_synseg[padded_idx][0]
        cellseg_data = src_cellseg[padded_idx][0]
        data_start = padded_idx.start

        # Iterate over the synapse IDs
        syn_ids = np.unique(synseg_data)
        syn_ids = syn_ids[syn_ids != 0]
        logger.info(f"Chunk {idx.chunk_id} contains {len(syn_ids)} synapses")
        # pylint: disable=consider-using-with

        line_id: int = idx.chunk_id * 10000

        line_annotations = []
        for syn_id in syn_ids:
            # Find centroid; if it's outside our index (including, in the padding area),
            # then skip it.
            centroid: Vec3D = Vec3D(*centroid_of_id(synseg_data, syn_id))
            logger.info(f"synapse {syn_id}  has centroid {centroid}")
            if not idx.contains(centroid + data_start):
                logger.info("not in central region -- skipping it")
                continue

            # Find the cell that most overlaps our synapse area (mask).
            # This identifies the cell on this side of the synapse.
            synapse_mask = synseg_data == syn_id
            source_cell_id = most_overlapping_id(cellseg_data, synapse_mask)
            if source_cell_id is None:
                logger.warning(
                    f"Synapse {syn_id} overlaps no cell at {tuple(data_start[0] + centroid)}"
                )
                continue

            if self.valid_source_ids and source_cell_id not in self.valid_source_ids:
                # This cell is not on our list of cells of interest, so skip this synapse.
                logger.info(f"cell {source_cell_id} is not on the whitelist -- skipping it")
                continue

            # Dilate the synapse (or ribbon -- whatever it may be) by the given distance.
            dilated_synapse_mask = fast_dilate(synapse_mask, self.max_distance, idx.resolution)

            # Now, find all segment IDs that overlap this dilated mask
            cellseg_masked = np.where(dilated_synapse_mask, cellseg_data, 0)
            nearby_segment_ids = [
                i for i in np.unique(cellseg_masked) if i > 0 and i != source_cell_id
            ]
            logger.info(
                f"found {len(nearby_segment_ids)} nearby segment IDs: {nearby_segment_ids}"
            )
            if len(nearby_segment_ids) == 0:
                continue

            # For the source point, we'll use the centroid of the whole synapse/ribbon
            # (or the nearest valid point to that).  This will produce a "starburst" of
            # connections which all come from a single point.
            source_point = nearest_value(cellseg_data, centroid, source_cell_id, True)
            if source_point is None:
                source_point = nearest_value(cellseg_data, centroid, source_cell_id, False)
            source_point = round(Vec3D(*source_point))

            # Finding a good destination point is a bit of a chore, so we'll factor
            # that out into a helper method.
            for dest_cell_id in nearby_segment_ids:
                # Create an annotation for the precomputed lines file.
                dest_point = find_valid_point(cellseg_masked, dest_cell_id, source_point)
                if dest_point is None:
                    logger.error(f"Couldn't find a valid point for {dest_cell_id}!")
                    continue
                final_id = cellseg_data[dest_point[0], dest_point[1], dest_point[2]]
                if final_id != dest_cell_id:
                    # This should not happen
                    logger.error(
                        f"Chosen point {dest_point} is cell {final_id} "
                        f"rather than {dest_cell_id} - DOH!!!"
                    )

                line_annotations.append(
                    LineAnnotation(
                        data_start + source_point, data_start + Vec3D(*dest_point), id=line_id
                    )
                )
                line_id += 1
                logger.info(line_annotations[-1])

        # write precomputed annotations to smallest (lowest-level) chunks ONLY
        dst_lines.write_annotations(line_annotations, all_levels=False)


def find_valid_point(
    masked_volume: np.ndarray, cell_id: int, nearby_point: Tuple[int, int, int]
) -> Optional[Tuple[int, int, int]]:
    """
    Find a good point in masked_volume to represent the given id.

    :param masked_volume: 3D array (X, Y, Z) of cell IDs within the region of interest.
    :param cell_id: Cell ID to find point for.
    :param nearby_point:  (x,y,z) coordinate to stay close to.
    :return: A coordinate tuple (x, y, z) if a valid point is found, otherwise None.
    """
    # Find a good Z slice to use.
    # Weigh Z-slices by both presence of id and proximity to target Z
    id_mask = masked_volume == cell_id
    z_sums = np.sum(id_mask, axis=(0, 1))
    if not z_sums.any():
        return None

    z_scores = z_sums / z_sums.max()  # Normalize to [0,1]
    z_dists = 1 - np.abs(np.arange(z_sums.shape[0]) - nearby_point[2]) / masked_volume.shape[2]
    best_z = np.argmax(z_scores + z_dists)
    if z_sums[best_z] == 0:
        best_z = np.argmax(z_sums)

    # Get most interior point at chosen Z
    slice_mask = masked_volume[:, :, best_z]
    if not slice_mask.any():
        return None
    binary_mask = slice_mask == cell_id
    dist_transform = ndimage.distance_transform_edt(binary_mask)
    # pylint: disable=unbalanced-tuple-unpacking
    x, y = np.unravel_index(np.argmax(dist_transform), slice_mask.shape)

    return (int(x), int(y), int(best_z))


def are_segments_touching(volume, id1, id2):
    """
    Check if two segments in a 3D volume are touching or adjacent.

    :param volume: 3D array where values represent segment IDs.
    :param id1: First segment ID to check for adjacency.
    :param id2: Second segment ID to check for adjacency.
    :return: True if the segments are touching, False otherwise.
    """
    mask1 = volume == id1
    mask2 = volume == id2
    dilated_mask1 = ndimage.binary_dilation(mask1, iterations=3)
    # ToDo: consider using fast_dilate instead
    # (need to measure and see which is faster in this case)
    return np.any(dilated_mask1 & mask2)


def get_segment_volumes(volume):
    """
    Calculate the volumes (voxel counts) for all segments in the volume.

    :param volume: 3D array where values represent segment IDs.
    :return: Dictionary mapping segment IDs to their volumes (voxel counts).
    """
    unique, counts = np.unique(volume, return_counts=True)
    return dict(zip(unique, counts))


def find_largest_touching_triad(volume, base_id):
    """
    Find the two largest segments that both touch the base segment and each other.

    :param volume: 3D array where values represent segment IDs.
    :param base_id: ID of the base segment to find touching pairs for.
    :return: A tuple (id1, id2, volume1, volume2) representing the largest touching pair,
             or None if no triad exists. ``id1`` and ``id2`` are the segment IDs, and
             ``volume1`` and ``volume2`` are their respective volumes.
    """
    # Get all segment volumes
    volumes = get_segment_volumes(volume)

    # Find all segments that touch the base segment
    touching_base = []
    for segment_id in volumes.keys():  # pylint: disable=consider-using-dict-items
        if segment_id == base_id:
            continue
        if are_segments_touching(volume, base_id, segment_id):
            touching_base.append((segment_id, volumes[segment_id]))

    # Sort segments by volume (largest first)
    touching_base.sort(key=lambda x: x[1], reverse=True)

    # Find the largest pair that also touches each other
    best_pair = (None, None, 0, 0)
    best_volume = 0

    # Check pairs of touching segments
    for i, (id1, vol1) in enumerate(touching_base[:-1]):
        for id2, vol2 in touching_base[i + 1 :]:
            if are_segments_touching(volume, id1, id2):
                current_volume = vol1 + vol2
                if current_volume > best_volume:
                    best_volume = current_volume
                    best_pair = (id1, id2, vol1, vol2)

    if best_pair[0] is None and touching_base:
        # No triad found?  Return a single synapse instead
        idnum, vol = touching_base[0]
        return (idnum, None, vol, None)

    return best_pair


@builder.register("AssignTriadsOp")
@mazepa.taskable_operation_cls
@attrs.frozen()
class AssignTriadsOp:  # implements VolumetricOpProtocol
    crop_pad: Sequence[int] = (0, 0, 0)

    # We assign synapses to all cells within this distance (in XY pixels) of the
    # centroid of each input cluster (except for the cell containing that centroid).
    # (Note that padding of at least this size should also be applied.)
    max_distance: float = 20.0

    # How far (in XY pixels) to shift the endpoint of the line further beyond
    # the nearest point in the target cell, just to make it easier to see that
    # it really is within that cell, and not just sitting on the border.
    endpoint_shift: float = 4

    # If not None, this is a list of cell IDs which are acceptable cell
    # segment IDs for the source end of each synapse; any syn segments
    # that map to a cell not in this list will be ignored.
    valid_source_ids: Sequence[int] | None = None

    # pylint: disable=no-self-use
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        # For simplicity, just return the destination resolution
        return dst_resolution

    # pylint: disable=unused-argument
    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "AssignTriadsOp":
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    # pylint: disable=redefined-outer-name
    def __call__(
        self,
        idx: VolumetricIndex,
        dst_lines: AnnotationLayerBackend,
        src_synseg: VolumetricLayer,
        src_cellseg: VolumetricLayer,
        *args,
        **kwargs,
    ):
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        # logger.info(
        #     f"Op called with dst_lines: {dst_lines}, idx: {idx}, crop_pad: {self.crop_pad}"
        # )

        # Load data from our two input layers, into numpy arrays
        padded_idx = idx.padded(self.crop_pad)
        synseg_data = src_synseg[padded_idx][0]
        cellseg_data = src_cellseg[padded_idx][0]
        data_start = padded_idx.start

        # Iterate over the synapse IDs
        syn_ids = np.unique(synseg_data)
        syn_ids = syn_ids[syn_ids != 0]
        logger.info(f"Chunk {idx.chunk_id} contains {len(syn_ids)} synapses")
        # pylint: disable=consider-using-with
        max_dist_squared = self.max_distance ** 2

        line_id: int = idx.chunk_id * 10000

        # csv_file.write(f"chunk {idx.chunk_id} contains {len(syn_ids)} ribbons: {syn_ids}\n")
        line_annotations = []
        for syn_id in syn_ids:
            # Find centroid; if it's outside our index (including, in the padding area),
            # then skip it.
            centroid: Vec3D = Vec3D(*centroid_of_id(synseg_data, syn_id))
            if not idx.contains(centroid + data_start):
                continue

            # Find the Z-slice with the maximum amount of syn_id in it.
            # best_z = np.argmax(np.sum(synseg_data == syn_id, axis=(0, 1)))
            # ....or, not; instead let's just use the centroid Z.
            best_z = round(centroid.z)
            synseg_2d = synseg_data[:, :, best_z]
            cellseg_2d = cellseg_data[:, :, best_z]

            # Find the cell that most overlaps our synapse area (mask).
            # This identifies the cell on this side of the synapse.
            synapse_mask = synseg_2d == syn_id
            source_cell_id = most_overlapping_id(cellseg_2d, synapse_mask)
            if source_cell_id is None:
                logger.warning(
                    f"Synapse {syn_id} overlaps no cell at {tuple(data_start[0] + centroid)}"
                )
                continue

            if self.valid_source_ids and source_cell_id not in self.valid_source_ids:
                # This cell is not on our list of cells of interest, so skip this synapse.
                continue

            # Redefine the centroid using this 2D slice.  This will be our starting
            # point for finding the neighboring cells.
            x_coords, y_coords = np.nonzero(synapse_mask)
            centroid = Vec3D(round(np.mean(x_coords)), round(np.mean(y_coords)), best_z)

            # Now, find the best triad touching syn_id.
            x, y = np.ogrid[: cellseg_2d.shape[0], : cellseg_2d.shape[1]]
            dist_squared = (x - centroid[0]) ** 2 + (y - centroid[1]) ** 2
            circle_mask = dist_squared <= max_dist_squared
            cellseg_masked = cellseg_2d[circle_mask]
            id1, id2, vol1, vol2 = find_largest_touching_triad(  # pylint: disable=unused-variable
                cellseg_masked, source_cell_id
            )

            for dest_cell_id in (id1, id2):
                if dest_cell_id is None:
                    continue
                # Create an annotation for the precomputed lines file.
                # We'll use the centroid on one side, and the nearest point in the
                # neighboring cell for the other side (extended a few pixels to make
                # it easier to see, and more robust to cell segmentation tweaks.)
                source_point = bfs_nearest_value(cellseg_data, centroid, source_cell_id, True)
                if source_point is None:
                    source_point = bfs_nearest_value(cellseg_data, centroid, source_cell_id, False)
                source_point = Vec3D(*source_point) + data_start

                dest_point = nearest_value(cellseg_data, centroid, dest_cell_id, True)
                if dest_point is None:
                    dest_point = nearest_value(cellseg_data, centroid, dest_cell_id, False)
                if dest_point is None:
                    logger.error(f"WTF?!? Couldn't find {dest_cell_id} near {centroid}")
                    continue
                dest_point = Vec3D(*dest_point) + data_start
                dest_point += normalized(dest_point - source_point) * self.endpoint_shift
                line_annotations.append(LineAnnotation(source_point, dest_point, id=line_id))
                line_id += 1
                # logger.info(line_annotations[-1])

        # write precomputed annotations to smallest (lowest-level) chunks ONLY
        dst_lines.write_annotations(line_annotations, all_levels=False)
