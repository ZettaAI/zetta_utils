"""
This module provides CCEdgeClear, a volumetric operation that performs
Connected Components (using cc3d), followed by clearing (zeroing out)
any segments that touch the edge of each chunk.

It also assigns each segment a unique ID based on the chunk ID (based
on an adjustable number of ids per chunk).

This is useful, in combination with the 'max' blend mode, in cases where
no segment is expected to be larger than the overlap region.  In this
case, any segment that is zeroed out by one chunk will be intact (and
nonzero) in the neighboring chunk.  The max blend mode will reconcile
these into a coherent set of unique IDs for the entire volume.

For example, this can be used to segment synapses.

Example CUE file:
----------------------------------------------------------------------
#SRC_PATH: "gs://dkronauer-ant-001-synapse/test/inference20240329023545"
#DST_PATH: "gs://dkronauer-ant-001-synapse/test/synseg20240513a"
#RESOLUTION: [16, 16, 42]

#BBOX: {
	"@type": "BBox3D.from_coords"
    start_coord: [14250, 9800, 3060]
    end_coord: [14762, 10312, 3092]
    resolution: #RESOLUTION
}

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX
	dst_resolution: #RESOLUTION
	processing_chunk_sizes: [[512, 512, 32], [320, 320, 32]]
	processing_blend_pads: [[64, 64, 0], [64, 64, 0]]
	processing_blend_modes: ["max", "max"]
	max_reduction_chunk_size: [512, 512, 40]
    skip_intermediaries: false
    level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/", "file://~/.zetta_utils/tmp/"]
	expand_bbox_processing: true

	// Specification for the operation we're performing
	op: {
		"@type":    "CCEdgeClear"
	}
	// Specification for the input (source) for the operation
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}

	// Specification of the output layer.
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
        // define our destination layer by starting with the source info...
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
        // but override the layer type to 1 channel of int64 data...
        info_field_overrides: {
            num_channels: 1
            data_type:    "int32"
        }
        // and clear out all the unused scales, which otherwise make NG really hard to use
        info_add_scales: [dst_resolution]
        info_add_scales_mode: "replace"
		cv_kwargs: {
			non_aligned_writes: true
		}
	}
}

// Execution parameters
"@type":                "mazepa.execute"
target:  #FLOW
----------------------------------------------------------------------
Keywords: segmentation, synapses, connected components
"""
from typing import Sequence

import attrs
import cc3d
import numpy as np
import torch

from zetta_utils import builder, log, mazepa
from zetta_utils.geometry import Vec3D
from zetta_utils.layer.volumetric import VolumetricIndex, VolumetricLayer
from zetta_utils.tensor_ops import convert, crop

logger = log.get_logger("zetta_utils")


@builder.register("CCEdgeClear")
@mazepa.taskable_operation_cls
@attrs.frozen()
class CCEdgeClear:  # implements VolumetricOpProtocol
    """
    VolumetricOpProtocol that performs Connected Components, followed
    by clearing (setting to 0) any clusters that touch the chunk edge.

    It also assigns each segment a unique ID based on the chunk ID
    (times ids_per_chunk, which should be larger than the number of
    segments found in any single chunk).
    """

    threshold: float = 0.1
    connectivity: int = 6
    ids_per_chunk: int = 10000
    crop_pad: Sequence[int] = (0, 0, 0)

    # pylint: disable=R6301
    def get_input_resolution(self, dst_resolution: Vec3D) -> Vec3D:
        # For simplicity, just return the destination resolution
        return dst_resolution

    def with_added_crop_pad(self, crop_pad: Vec3D[int]) -> "CCEdgeClear":
        return attrs.evolve(self, crop_pad=Vec3D[int](*self.crop_pad) + crop_pad)

    def __call__(
        self, idx: VolumetricIndex, dst: VolumetricLayer, src: VolumetricLayer, *args, **kwargs
    ):
        logger.info(
            f">>>>> CCEdgeClear called for chunk_id {idx.chunk_id}, bounds {idx.bbox.bounds},"
            f" data shape {src[idx].shape}, with crop_pad {self.crop_pad}"
        )
        # Use cc3d to get local segment labels, starting at 1 (these are stored as uint16).
        data_np = convert.to_np(src[idx.padded(self.crop_pad)][0])  # (use channel 0)
        cc_labels = cc3d.connected_components(
            data_np > self.threshold, connectivity=self.connectivity
        )
        count = cc_labels.max()  # because cluster IDs count sequentially from 1 at this stage
        logger.info(f">>>>>> Found {count} clusters")
        assert count < self.ids_per_chunk, (
            f"Cluster count {count} exceeds ids_per_chunk {self.ids_per_chunk}"
            f" in chunk {idx.chunk_id}"
        )

        # ToDo: zero out any clusters below size threshold.

        # Zero out any clusters that touch any face of the chunk.
        edge_cluster_ids = np.unique(
            np.concatenate(
                (
                    cc_labels[0, :, :].flatten(),
                    cc_labels[-1, :, :].flatten(),
                    cc_labels[:, 0, :].flatten(),
                    cc_labels[:, -1, :].flatten(),
                    cc_labels[:, :, 0].flatten(),
                    cc_labels[:, :, -1].flatten(),
                )
            )
        )
        edge_cluster_ids = edge_cluster_ids[edge_cluster_ids != 0]
        logger.info(f"Clearing {len(edge_cluster_ids)} edge IDs")
        if len(edge_cluster_ids) > 0:
            mask = np.isin(cc_labels, edge_cluster_ids)
            cc_labels[mask] = 0

        # Convert to pytorch longs, and offset nonzero values by some factor of the chunk size.
        offset = self.ids_per_chunk * idx.chunk_id
        labels_tensor = torch.tensor(
            cc_labels.astype(np.int32)
        )  # convert labels to pytorch-compatible type
        nonzero_indices = labels_tensor != 0
        if nonzero_indices.any():
            labels_tensor[nonzero_indices] += offset
            logger.info(
                f"As this is chunk {idx.chunk_id}, remapped IDs to "
                f"{labels_tensor[nonzero_indices].min()} - {labels_tensor.max()}"
            )
        else:
            logger.info("This leaves 0 labels in this chunk")

        labels_tensor = crop(labels_tensor, self.crop_pad)
        dst[idx] = labels_tensor.unsqueeze(0)  # convert back to CXYZ
