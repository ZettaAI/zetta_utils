"""
This script provides (and runs) a custom VolumetricOp that does synapse
assignment via an assignment model.
"""
from collections import deque
from math import floor
from typing import Sequence

import attrs
import numpy as np
import torch
from scipy.stats import mode

import zetta_utils
import zetta_utils.mazepa_layer_processing.common
from zetta_utils import builder, log, mazepa
from zetta_utils.convnet.utils import load_model
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer, VolumetricAnnotationLayer
from zetta_utils.layer.volumetric.annotation.backend import (
    AnnotationLayerBackend,
    LineAnnotation,
)

logger = log.get_logger("zetta_utils")


def centroid_of_id(array: np.ndarray, id_value: int):
    """
    Find the (rounded to int) centroid of locations in the array with a value equal to id.
    """
    coordinates = np.argwhere(array == id_value)
    assert coordinates.size > 0
    centroid = np.mean(coordinates, axis=0)
    return np.round(centroid).astype(int)


def bfs_nearest_value(
    array: np.ndarray, start: Sequence[int], target_value: int, xy_only: bool = False
):
    """
    Perform a breadth-first search starting at the given point, to find
    the closest point in the array with the given value.

    Returns: (x, y, z) location of the point found, or None.
    """
    start = tuple(round(value) for value in start)
    # print(f'bfs_nearest_value starting at {start}, looking for {target_value}')
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    if not xy_only:
        directions += [(0, 0, 1), (0, 0, -1)]
    queue = deque([start])
    visited = set()
    visited.add(tuple(start))

    while queue:
        x, y, z = queue.popleft()
        if array[x, y, z] == target_value:
            return (x, y, z)  # Found it!
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < array.shape[0] and 0 <= ny < array.shape[1] and 0 <= nz < array.shape[2]:
                neighbor_pos = (nx, ny, nz)
                if neighbor_pos not in visited:
                    queue.append(neighbor_pos)
                    visited.add(neighbor_pos)
    return None  # None found


@builder.register("AssignSynapsesOp")
@mazepa.taskable_operation_cls
@attrs.frozen()
class AssignSynapsesOp:  # implements VolumetricOpProtocol
    model_path: str
    crop_pad: Sequence[int] = (0, 0, 0)
    window_size: Sequence[int] = (24, 24, 8)

    # which output channel (0 or 1) represents the desired synaptic partner:
    partner_channel: int = 0

    # pylint: disable=no-self-use
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        # For simplicity, just return the destination resolution
        return dst_resolution

    # pylint: disable=unused-argument
    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "AssignSynapsesOp":
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    # pylint: disable=redefined-outer-name
    def __call__(
        self,
        idx: VolumetricIndex,
        dst: VolumetricLayer,
        dst_lines: VolumetricAnnotationLayer,
        src_image: VolumetricLayer,
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
        assert isinstance(dst_lines, VolumetricAnnotationLayer), f"dst_lines must be a VolumetricAnnotationLayer, not {type(dst_lines)}"

        # Load assignment model
        model = load_model(self.model_path)
        assert model is not None, f"Unable to load model at {self.model_path}"
        # model_input = model.get_inputs()[0]
        # input_name = model_input.name
        # input_shape = model_input.shape
        # window_size = [input_shape[4], input_shape[3], input_shape[2]]
        # logger.info(
        #     f"Model input shape: {input_shape}; Window size: {window_size}; "
        #     f"type {model_input.type}"
        # )

        # Load data from our three input layers, into numpy arrays
        padded_idx = idx.padded(self.crop_pad)
        image_data = src_image[padded_idx][0]
        logger.info(f"Got image data of shape: {image_data.shape}")
        synseg_data = src_synseg[padded_idx][0]
        logger.info(f"Got synseg data of shape: {synseg_data.shape}")
        cellseg_data = src_cellseg[padded_idx][0]
        logger.info(f"Got cellseg data of shape: {cellseg_data.shape}")
        dest_data = np.zeros(image_data.shape)
        data_start = idx.start - Vec3D(*self.crop_pad)

        # Iterate over the synapse IDs
        syn_ids = np.unique(synseg_data)
        syn_ids = syn_ids[syn_ids != 0]
        logger.info(f"Chunk {idx.chunk_id} contains {len(syn_ids)} synapses")
        # pylint: disable=consider-using-with
        with open(f"syn_inf_{idx.chunk_id}.csv", "w", encoding="utf-8") as csv_file:
            line_annotations = []
            for syn_id in syn_ids:
                centroid = centroid_of_id(synseg_data, syn_id)

                # Extract small window around the synapse in our various layers.
                xmin = floor(centroid[0] - self.window_size[0] / 2)
                ymin = floor(centroid[1] - self.window_size[1] / 2)
                zmin = floor(centroid[2] - self.window_size[2] / 2)
                xmax = xmin + self.window_size[0]
                ymax = ymin + self.window_size[1]
                zmax = zmin + self.window_size[2]
                # pylint: disable=too-many-boolean-expressions
                if (
                    xmin < 0
                    or ymin < 0
                    or zmin < 0
                    or xmax >= image_data.shape[0]
                    or ymax >= image_data.shape[1]
                    or zmax >= image_data.shape[2]
                ):
                    logger.info("Window is out of bounds; skipping this one.")
                    continue
                image_wind = image_data[xmin:xmax, ymin:ymax, zmin:zmax]
                synseg_wind = synseg_data[xmin:xmax, ymin:ymax, zmin:zmax]
                cellseg_wind = cellseg_data[xmin:xmax, ymin:ymax, zmin:zmax]

                # Run the model on this window.
                syn_mask = np.where(synseg_wind == syn_id, 1, 0)
                input_tensor = np.stack(
                    [  # Reshape inputs (8, 24, 24) -> (2, 8, 24, 24)
                        np.transpose(image_wind, (2, 0, 1)),
                        np.transpose(syn_mask, (2, 0, 1)),
                    ]
                )
                input_tensor = np.expand_dims(
                    input_tensor, axis=0
                )  # Add batch dimension to the input tensor
                input_tensor = input_tensor.astype(np.float32)

                output_array = model(torch.from_numpy(input_tensor)).detach().numpy()

                # transpose to match the pattern of our image data
                # presyn_output = np.transpose(output_array[0, 0], (1, 2, 0))
                # postsyn_output = np.transpose(output_array[0, 1], (1, 2, 0))
                partner_output = np.transpose(output_array[0, self.partner_channel], (1, 2, 0))

                # (Add to output data, just for debugging purposes)
                dest_data[xmin:xmax, ymin:ymax, zmin:zmax] += partner_output

                # 6. Find the cell ID that maximizes the presynaptic output.
                best_sum = 0
                pre_cell_id: int = 0
                candidate_cell_ids = np.unique(
                    cellseg_wind
                )  # (or could be limited by dilated synapse mask)
                for cell_id in candidate_cell_ids:
                    cell_mask = np.where(cellseg_wind == cell_id, 1, 0)
                    total = np.sum(cell_mask * partner_output)
                    if total > best_sum:
                        best_sum = total
                        pre_cell_id = cell_id

                # ...and to find the "self" cell ID, just take the most common value
                # (i.e. mode) in the cell ID window under the synapse mask.
                self_cell_id = mode(cellseg_wind[syn_mask.astype(bool)])[0]
                logger.info(f"Synapse: {syn_id}  Other: {pre_cell_id}  Self: {self_cell_id}")

                # Write to CSV
                csv_file.write(f"{syn_id},{pre_cell_id},{self_cell_id}\n")

                # Also create an annotation for the precomputed lines file.
                # NOTE: variable names below assume that the input is the "post" side
                # and the partner is the "pre" side.  The opposite may be true, depending
                # on how the models were trained and which partner_channel was chosen.
                window_center = tuple(i // 2 for i in self.window_size)
                pre_cell_point = bfs_nearest_value(cellseg_wind, window_center, pre_cell_id, True)
                if pre_cell_point is None:
                    pre_cell_point = bfs_nearest_value(
                        cellseg_wind, window_center, pre_cell_id, False
                    )
                pre_cell_point = Vec3D(*pre_cell_point) + data_start + Vec3D(xmin, ymin, zmin)
                post_cell_point: Vec3D = Vec3D(*centroid) + data_start
                line_annotations.append(LineAnnotation(start=post_cell_point, end=pre_cell_point, id=syn_id))
                logger.info(line_annotations[-1])
                print(line_annotations[-1])

        # write precomputed annotations to smallest (lowest-level) chunks ONLY
        logger.info("Writing line_annotations")
        dst_lines.backend.write_annotations(line_annotations, all_levels=False)
        logger.info("Done writing line_annotations")
        # write presynaptic predictions to volumetric image layer

        x_pad, y_pad, z_pad = self.crop_pad
        if x_pad or y_pad or z_pad:
            dest_data = dest_data[
                x_pad : -x_pad if x_pad > 0 else None,
                y_pad : -y_pad if y_pad > 0 else None,
                z_pad : -z_pad if z_pad > 0 else None,
            ]
        dst[idx] = np.expand_dims(dest_data, axis=0).astype(np.float32)
        logger.info("Done writing to dst[idx]")


if __name__ == "__main__":
    # spec = zetta_utils.parsing.cue.load("~/joes_test_code/zheng_hippocampus/zheng_testcut_asn.cue")
    # flow = builder.build(spec["target"])
    # mazepa.execute(flow)
    # logger.info("Flow complete.")

    spec = zetta_utils.parsing.cue.load("../../../specs/joe/syn_assignment.cue")
    target = spec["target"]

    # Start by preparing the output annotation file.  This step (creating the info file)
    # should only be done once, not by each chunk.
    dst_lines_spec = target["op_kwargs"]["dst_lines"]
    dst_lines = builder.build(dst_lines_spec)
    logger.info(f"Clearing spatial file: {dst_lines}")
    dst_lines.clear()

    flow = builder.build(spec["target"])
    mazepa.execute(flow)
    logger.info("Flow complete.")

    # Rewrite the info file with correct counts.
    dst_lines.post_process()

    logger.info("Run complete.")
    logger.info(f"Line annotations written to: {dst_lines.path}")
