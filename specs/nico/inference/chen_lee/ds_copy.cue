import "list"
import "math"

#DST_IMG_PATH: "gs://zetta_jchen_mouse_cortex_001_raw/unaligned/img"

#REFERENCE_RES: [3.75, 3.75, 50]
#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],
	// [1344 * #REFERENCE_RES[2], 1366 * #REFERENCE_RES[2]],
	// [1366 * #REFERENCE_RES[2], 3749 * #REFERENCE_RES[2]],
	// [1361 * #REFERENCE_RES[2], 3761 * #REFERENCE_RES[2]],
	// [2545 * #REFERENCE_RES[2], 2897 * #REFERENCE_RES[2]],
	[2561 * #REFERENCE_RES[2], 2577 * #REFERENCE_RES[2]],
]

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240320_2"
worker_resource_requests: {
	memory:           "39560Mi"
}
worker_replicas:      20
batch_gap_sleep_sec:  0.1
do_dryrun_estimation: true
local_test:           true
debug: true
worker_cluster_project: "jchen-mouse-cortex-001"
worker_cluster_region: "us-east1"
worker_cluster_name: "zutils"

num_procs: 6
semaphores_spec: {
	read:  6
	write: 6
	cuda:  1
	cpu:   1
}

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		for factor in [64, 128, 256] {
			#DOWNSAMPLE_IMG_TEMPLATE & {
				_src_path: #DST_IMG_PATH
				_dst_path: #DST_IMG_PATH
				dst_resolution: [#REFERENCE_RES[0] * factor, #REFERENCE_RES[1] * factor, #REFERENCE_RES[2]]
			},
		}
	]
}

#MAX_TASK_SIZE: [8192, 8192, 16]
#DOWNSAMPLE_IMG_TEMPLATE: {
	_src_path:     string
	_dst_path:     string

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/dst_resolution[1])]),
		16,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "InterpolateOperation"
		mode:    "img"
		res_change_mult: [2, 2, 1]
	}
	dst_resolution: [number, number, number]
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    _src_path
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _dst_path
		// info_reference_path: _src_path
		// info_add_scales:     [{"resolution": dst_resolution, "encoding": "jpeg", chunk_sizes: [[256,256,16]]}]
		// info_add_scales_mode: "merge"
		cv_kwargs: {"compress": false, "cache": false}
		// on_info_exists:       "overwrite"
	}
}
