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

_ALIGN_PARAMS: {
	"24576": {channel: -1, sm: 100, num_iter: 700, lr: 0.001, src_zeros_sm_mult: 1.0, tgt_zeros_sm_mult: 1.0},
	"12288": {channel: 1, sm: 500, num_iter: 700, lr: 0.001, src_zeros_sm_mult: 0.001, tgt_zeros_sm_mult: 0.001},
	"6144": {channel: 0, sm: 500, num_iter: 700, lr: 0.0025, src_zeros_sm_mult: 0.001, tgt_zeros_sm_mult: 0.001},
}


// INPUT
// #UNALIGNED_IMG_PATH:    "https://td.princeton.edu/sseung-test1/ca3-alignment-temp/full_section_imap4"
#UNALIGNED_ENC_PATH:    "gs://zetta-research-nico/hippocampus/low_res_enc_c4"
#RIGID_ENC_PATH:        "gs://zetta-research-nico/hippocampus/rigid_w_scale/low_res_enc_c4_rigid"
#AFFINE_ENC_PATH:       "gs://zetta-research-nico/hippocampus/pairwise/affine_100th_w_scale_inv/low_res_enc_c4"
#AFFINE_FIELD_PATH:     "gs://zetta-research-nico/hippocampus/pairwise/affine_100th_w_scale/field"
#AFFINE_INV_FIELD_PATH:     "gs://zetta-research-nico/hippocampus/pairwise/affine_100th_w_scale_inv/field"

// OUTPUT
#COARSE_WO_AFFINE_FIELD_PATH: "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_wo_affine/field"
#COARSE_WO_AFFINE_INV_FIELD_PATH: "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_wo_affine_inv/field"
#COARSE_FIELD_PATH:     "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale/field"
#COARSE_INV_FIELD_PATH: "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_inv/field"
#COARSE_ENC_PATH:       "gs://zetta-research-nico/hippocampus/pairwise/coarse_100th_w_scale_inv/low_res_enc_c4"

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
	"@type": "mazepa.concurrent_flow"
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
				#COMPUTE_FIELD_TEMPLATE & {
					_z_offset: tgt_z - z
					_bounds: bounds
				},
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
				#WARP_ENC_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [12288, 12288, 45]
				}
			]
		},

	]
}


#STAGE_TMPL: {
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/dst_resolution[1])]),
		1,
	]
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	fn: {
		"@type":  "align_with_online_finetuner"
		"@mode":  "partial"
		channel:  int
		sm:       int
		num_iter: int
		lr:       float
		src_zeros_sm_mult: number
		tgt_zeros_sm_mult: number
	}
}

#COMPUTE_FIELD_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	_z_offset: int

	"@type": "build_compute_field_multistage_flow"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	stages: [
		// #STAGE_TMPL & {  // 24576
		// 	dst_resolution: [24576, 24576, #REFERENCE_RES[2]]
		// 	fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["24576"]
		// },
		#STAGE_TMPL & {  // 12288
			dst_resolution: [12288, 12288, #REFERENCE_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["12288"]
		},
		#STAGE_TMPL & {  // 6144
			dst_resolution: [6144, 6144, #REFERENCE_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["6144"]
		},
	]

	tgt_offset: [0, 0, _z_offset]
	offset_resolution: #REFERENCE_RES

	src: {
		"@type": "build_cv_layer"
		path:    #AFFINE_ENC_PATH
		read_procs: [
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[1:2,:,:,:] == -103, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == -120, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == 112, 0, x)"}
		]
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #RIGID_ENC_PATH
		read_procs: [
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[1:2,:,:,:] == -103, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == -120, 0, x)"},
			// {"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == 112, 0, x)"}
		]
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #COARSE_WO_AFFINE_FIELD_PATH
		info_reference_path: #AFFINE_FIELD_PATH
		info_add_scales: [[12288, 12288, 45], [24576, 24576, 45]]
		info_add_scales_mode: "merge"
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: #COARSE_WO_AFFINE_FIELD_PATH + "/tmp"
	tmp_layer_factory: {
		"@type":              "build_cv_layer"
		"@mode":              "partial"
		info_reference_path:  #AFFINE_FIELD_PATH
		info_add_scales:     [[12288, 12288, 45], [24576, 24576, 45]]
		info_add_scales_mode: "merge"
		on_info_exists:       "overwrite"
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
		crop_pad: [256, 256, 0]
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
			path:    #AFFINE_INV_FIELD_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #COARSE_WO_AFFINE_FIELD_PATH
			// data_resolution: [6144, 6144, 45]
			// interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_INV_FIELD_PATH
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
			path:    #COARSE_INV_FIELD_PATH
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_FIELD_PATH
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
			path:    #UNALIGNED_ENC_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #COARSE_INV_FIELD_PATH
			data_resolution: [6144, 6144, 45]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COARSE_ENC_PATH
		info_reference_path: op_kwargs.src.path
		info_add_scales:     [[6144, 6144, 45], [12288, 12288, 45]]
		info_add_scales_mode: "replace"
		on_info_exists:      "overwrite"
	}
}
