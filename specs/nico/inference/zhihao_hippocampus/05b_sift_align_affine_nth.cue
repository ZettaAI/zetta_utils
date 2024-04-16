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

// INPUT
#UNALIGNED_ENC_PATH:                "gs://zetta-research-nico/hippocampus/low_res_enc_c4"
// #RIGID_FIELD_PATH:                  "gs://zetta-research-nico/hippocampus/rigid/field"
#RIGID_ENC_PATH:                    "gs://zetta-research-nico/hippocampus/rigid_w_scale/low_res_enc_c4_rigid"

// OUTPUT
#AFFINE_TRANSFORM_PATH_100TH:             "gs://zetta-research-nico/hippocampus/affine_100th_w_scale/transform_LMedS"
#AFFINE_FIELD_PATH_100TH:                 "gs://zetta-research-nico/hippocampus/pairwise/affine_100th_w_scale/field"
#AFFINE_INV_FIELD_PATH_100TH:             "gs://zetta-research-nico/hippocampus/pairwise/affine_100th_w_scale_inv/field"
#AFFINE_ENC_PATH_100TH:                   "gs://zetta-research-nico/hippocampus/pairwise/affine_100th_w_scale_inv/low_res_enc_c4"

// #PAIRWISE_FWD_FIELD_PATH:           "gs://zetta-research-nico/hippocampus/pairwise/coarse/field/fwd"
// #PAIRWISE_INV_FIELD_PATH:           "gs://zetta-research-nico/hippocampus/pairwise/coarse/field/inv"
// #PAIRWISE_ENC_PATH:                 "gs://zetta-research-nico/hippocampus/pairwise/coarse/low_res_enc_c4"


#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]

#REFERENCE_RES: [3, 3, 45]

#DATASET_BOUNDS: [
    [0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],  // [0 * #REFERENCE_RES[0], 1474560 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],  // [0 * #REFERENCE_RES[1], 1474560 * #REFERENCE_RES[1]],
	[24 * #REFERENCE_RES[2], 3994 * #REFERENCE_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 524288 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 524288 * #REFERENCE_RES[1]],
	[24 * #REFERENCE_RES[2], 3994 * #REFERENCE_RES[2]],
]


"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240306"
// worker_resource_requests: {
// 	memory: "10000Mi"
// }
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
				#SIFT_TEMPLATE & {
					_z_offset: tgt_z - z
					_bounds: bounds
					dst_resolution: [6144, 6144, 45]
				},
				#GEN_AFFINE_TEMPLATE & {
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
					dst_resolution: [6144*2, 6144*2, 45]
				},
				#WARP_ENC_TEMPLATE & {
					_bounds: bounds
					dst_resolution: [6144*4, 6144*4, 45]
				}
			]
		},

	]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#FIELD_INFO_OVERRIDE: {
	_dst_resolution: _
	type: "image"
	data_type: "float32",
	num_channels: 2,
	scales: [
		{
			let vx_res = _dst_resolution
			let ds_offset = [ for j in [0, 1, 2] {
				#DATASET_BOUNDS[j][0] / _dst_resolution[j]  // technically should be floor
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((#DATASET_BOUNDS[j][1] - #DATASET_BOUNDS[j][0]) / _dst_resolution[j])
			}]

			chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
			resolution:   vx_res
			encoding:     "zfpc"
			zfpc_correlated_dims: [true, true, false, false]
			zfpc_tolerance: 0.001953125
			key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
			voxel_offset: ds_offset
			size:         ds_size
		}
	],
	
}

#SIFT_TEMPLATE: {
	_bounds: _ | *#ROI_BOUNDS
	_z_offset: int

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1]-_bounds[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1]-_bounds[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":  "Transform2D"
			transformation_mode: "full_affine"
			estimate_mode: "lmeds"
			ratio_test_fraction: 0.65
			contrast_threshold: 0.04
			edge_threshold: 10
			sigma: 1.2
			num_octaves: 5
			num_min_matches: 4
			ensure_scale_boundaries:  [0.85, 1.2]
		}
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	skip_intermediaries:    true
	expand_bbox_processing: true
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_ENC_PATH
			read_procs: [
				{"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == 112, 0, x)"},
				{"@type": "torch.add", other: 128, "@mode": "partial"},
				{"@type": "to_uint8", "@mode": "partial"},
				{"@type": "lambda", "lambda_str": "lambda x: x[2:3,:,:,:]"}
			]
		}
		tgt: {
			"@type": "build_cv_layer"
			path:    #RIGID_ENC_PATH
			read_procs: [
				{"@type": "lambda", "lambda_str": "lambda x: torch.where(x[0:1,:,:,:] == 112, 0, x)"},
				{"@type": "torch.add", other: 128, "@mode": "partial"},
				{"@type": "to_uint8", "@mode": "partial"},
				{"@type": "lambda", "lambda_str": "lambda x: x[2:3,:,:,:]"}
			]
			index_procs: [{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, _z_offset]
				resolution: dst_resolution
			}]
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path: #AFFINE_TRANSFORM_PATH_100TH
		index_procs: [{
			"@type": "VolumetricIndexOverrider"
			override_offset: [null, null, null]
			override_size: [2, 3, 1]
			override_resolution: [null, null, null]
		}]
		info_field_overrides: {
			type:                "image"
			data_type:           "float32"
			num_channels:        1
			scales: [
				{
					chunk_sizes:  [[2, 3, 1]]
					resolution:   dst_resolution
					encoding:     "raw"
					key:          "\(dst_resolution[0])_\(dst_resolution[1])_\(dst_resolution[2])"
					voxel_offset: [0, 0, #DATASET_BOUNDS[2][0] / dst_resolution[2]]
					size:         [2, 3, (#DATASET_BOUNDS[2][1] - #DATASET_BOUNDS[2][0]) / dst_resolution[2]]
				},
			]
		}
		on_info_exists: "overwrite"
	}
}

#GEN_AFFINE_TEMPLATE: {
	_bounds: _
	let vx_res = dst_resolution
	let x_shape = math.Ceil(((_bounds[0][1] - _bounds[0][0]) / vx_res[0]))
	let y_shape = math.Ceil(((_bounds[1][1] - _bounds[1][0]) / vx_res[1]))
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "FieldFromTransform2D"
			shape: [x_shape, y_shape]
		}
		crop_pad: [0, 0, 0]
	}
	dst_resolution: _
	skip_intermediaries: true
	processing_chunk_sizes: [[x_shape, y_shape, 1]]
	processing_crop_pads:   [[0, 0, 0]]
	expand_bbox_resolution: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		mat: {
			"@type": "build_cv_layer"
			path:    #AFFINE_TRANSFORM_PATH_100TH
			index_procs: [{
				"@type": "VolumetricIndexOverrider"
				override_offset: [null, null, null]
				override_size: [2, 3, 1]
				override_resolution: [6144, 6144, 45]
			}]
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #AFFINE_FIELD_PATH_100TH
		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_dst_resolution: dst_resolution
		}
		on_info_exists:       "overwrite"
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
			path:    #AFFINE_FIELD_PATH_100TH
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFFINE_INV_FIELD_PATH_100TH
		info_reference_path: op_kwargs.src.path
		on_info_exists:       "overwrite"
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
			path:    #AFFINE_INV_FIELD_PATH_100TH
			data_resolution: [6144, 6144, 45]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFFINE_ENC_PATH_100TH
		info_reference_path: op_kwargs.src.path
		on_info_exists:      "overwrite"
	}
}
