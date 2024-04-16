import "math"

import "list"
import "strconv"

#Z_SKIP_MAP: {
	// "124": 24,
	// "224": 124,
	// "324": 224,
	// "424": 324,
	// "524": 424,
	// "624": 524,
	// "725": 624,
	// "825": 725,
	// "924": 825,
	// "1024": 924,
	// "1124": 1024,
	// "1224": 1124,
	// "1325": 1224,
	// "1424": 1325,
	// "1524": 1424,
	// "1624": 1524,
	// "1739": 1624,
	// "1836": 1739,
	// "1936": 1836,
	// "2036": 1936,
	// "2136": 2036,
	// "2236": 2136,
	"2740": 2236,
	// "2840": 2740,
	// "2940": 2840,
	// "3040": 2940,
	// "3140": 3040,
	// "3240": 3140,
	// "3340": 3240,
	// "3440": 3340,
	// "3540": 3440,
	// "3640": 3540,
	// "3740": 3640,
	// "3840": 3740,
	// "3940": 3840,
	// "3993": 3940,
}

// #UNALIGNED_IMG_PATH:    "https://td.princeton.edu/sseung-test1/ca3-alignment-temp/full_section_imap4"
// #UNALIGNED_ENC_PATH:    "gs://zetta-research-nico/hippocampus/low_res_enc_c4"

// INPUT
#RIGID_FIELD_PATH: "gs://zetta-research-nico/hippocampus/rigid_w_scale/field" //OK
#RIGID_ENC_PATH:        "gs://zetta-research-nico/hippocampus/rigid_w_scale/low_res_enc_c4_rigid"
#COARSE_FIELD_PATH:     "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale/field"
#COARSE_INV_FIELD_PATH: "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_inv/field"

// OUTPUT
#COARSE_WO_RIGID_FIELD_PATH: "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_wo_rigid/field"
#COARSE_WO_RIGID_INV_FIELD_PATH: "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_wo_rigid_inv/field"
// #RIGID_INVERSE_FIELD_PATH: "gs://zetta-research-nico/hippocampus/rigid_inv/field"
// #COARSE_ENC_PATH:       "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_inv/low_res_enc_c4"
#COARSE_ALT_ENC_PATH: "gs://tmp_2w/nico/hippocampus/pairwise/coarse_100th_w_scale_inv/low_res_enc_c4"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]
#REFERENCE_RES: [3, 3, 45]

#Z_RANGE: [2740, 2741]

#DATASET_BOUNDS: [
	[0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],  // [0 * #REFERENCE_RES[0], 1474560 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],  // [0 * #REFERENCE_RES[1], 1474560 * #REFERENCE_RES[1]],
	[24 * #REFERENCE_RES[2], 3994 * #REFERENCE_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],
	[#Z_RANGE[0] * #REFERENCE_RES[2], #Z_RANGE[1] * #REFERENCE_RES[2]],
]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240306"
worker_resources: {
	"nvidia.com/gpu": "1"
}
worker_replicas: 20
local_test:      true
debug: true

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		for key, tgt_z in #Z_SKIP_MAP {
			let z = strconv.Atoi(key)
			let bounds = [
				[#ROI_BOUNDS[0][0], #ROI_BOUNDS[0][1]],
				[#ROI_BOUNDS[1][0], #ROI_BOUNDS[1][1]],
				[z * #REFERENCE_RES[2], (z+1) * #REFERENCE_RES[2]],
			]
			"@type": "mazepa.sequential_flow"
			stages: [
				// #DOWNSAMPLE_FIELD_TEMPLATE & {
				// 	_bounds: bounds
				// 	_path: #RIGID_FIELD_PATH
				// 	src_resolution: [1536, 1536, 45]
				// },
				// #DOWNSAMPLE_FIELD_TEMPLATE & {
				// 	_bounds: bounds
				// 	_path: #RIGID_FIELD_PATH
				// 	src_resolution: [3072, 3072, 45]
				// },
				#COMPOSE_FIELD_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
				#INVERT_FIELD_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
				#WARP_ENC_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
			]
		},

	]
}


#DOWNSAMPLE_FIELD_TEMPLATE: {
	_path:     string
	_bounds: _ | *#ROI_BOUNDS

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_interpolate_flow"
	mode:    "field"
	src_resolution: [number, number, number]
	dst_resolution: [src_resolution[0] * 2, src_resolution[1] * 2, src_resolution[2]]
	chunk_size: max_chunk_size
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}

	src: {
		"@type": "build_cv_layer"
		path:    _path
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _path
		info_reference_path: _path
		info_add_scales:     [dst_resolution]
		info_add_scales_mode: "merge"
		on_info_exists:      "overwrite"
	}

}

#COMPOSE_FIELD_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "field"
		crop_pad: [0, 0, 0]
		use_translation_adjustment: false
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: false
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #COARSE_FIELD_PATH
			cv_kwargs: {"cache": false}
		}
		field: {
			"@type": "build_cv_layer"
			path:    #RIGID_FIELD_PATH
			cv_kwargs: {"cache": false}
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_WO_RIGID_FIELD_PATH
		info_reference_path: op_kwargs.src.path
		on_info_exists:      "overwrite"
	}
}


#INVERT_FIELD_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #COARSE_WO_RIGID_FIELD_PATH
			cv_kwargs: {"cache": false}
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_WO_RIGID_INV_FIELD_PATH
		info_reference_path: op_kwargs.src.path
		info_add_scales:     [[6144, 6144, 45]]
		info_add_scales_mode: "replace"
		on_info_exists:      "overwrite"
	}
}

#WARP_ENC_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "image"
		crop_pad: [256, 256, 0]
		use_translation_adjustment: false
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #RIGID_ENC_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #COARSE_WO_RIGID_INV_FIELD_PATH
			data_resolution: [6144, 6144, 45]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_ALT_ENC_PATH
		info_reference_path: op_kwargs.src.path
		info_add_scales:     [[6144, 6144, 45]]
		info_add_scales_mode: "replace"
		on_info_exists:      "overwrite"
	}
}
