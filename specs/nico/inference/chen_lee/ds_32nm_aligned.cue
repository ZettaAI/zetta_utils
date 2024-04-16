import "list"
import "math"

#SRC_IMG_PATH: "gs://zetta_jchen_mouse_cortex_001_drop/songyang_003/2402261055_sections1344_3748_0206_aligned/aligned_16_links"
#DST_IMG_PATH: "gs://zetta_jchen_mouse_cortex_001_drop/songyang_003/2402261055_sections1344_3748_0206_aligned/aligned_16_links"

#REFERENCE_RES: [3.75, 3.75, 50]
#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],
	// [1344 * #REFERENCE_RES[2], 3760 * #REFERENCE_RES[2]],
	[1344 * #REFERENCE_RES[2], 1967 * #REFERENCE_RES[2]],
]

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240320_2"
worker_resource_requests: {
	memory:           "39560Mi"
}
worker_replicas:      50
batch_gap_sleep_sec:  0.1
do_dryrun_estimation: true
local_test:           false
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
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z in [3301, 2797, 2799,2560,2563,2659,2677,2628,2571,2665,2671,2674,2884,2882,2880,2877,2680,2678,3294,3008,3055,3288] { //2560, 2563, 2573, 2680, 2799, 2880, 2882, 2877, 2884, 2671, 2571, 2665, 2575, 3314] {
			let bounds = [
				#ROI_BOUNDS[0],
				#ROI_BOUNDS[1],
				[z * #REFERENCE_RES[2], (z + 1) * #REFERENCE_RES[2]],
			]

			"@type": "mazepa.sequential_flow"
			stages: [
				for factor in [32] {
					#DOWNSAMPLE_IMG_TEMPLATE & {
						_bounds: bounds
						_src_path: #DST_IMG_PATH
						_dst_path: #DST_IMG_PATH
						dst_resolution: [#REFERENCE_RES[0] * factor, #REFERENCE_RES[1] * factor, #REFERENCE_RES[2]]
					},
				}
			]
		}
	]
}

#MAX_TASK_SIZE: [32768, 32768, 1]
#DOWNSAMPLE_IMG_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	_src_path:     string
	_dst_path:     string

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
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
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
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
		info_reference_path: _src_path
		info_add_scales:     [dst_resolution]
		info_add_scales_mode: "merge"
		on_info_exists:       "overwrite"
	}
}
