import "math"

import "list"

#Z_SKIP_MAP: {
	for z in list.Range(24, 3994, 1) {
		"\(z)": int | *(z-1)
	}
}

#Z_SKIP_MAP: {
	// START: 24
	"37": 33,
	"40": 38,
	"44": 42,
	"56": 54,
	"58": 56,
	"79": 67,
	"86": 84,
	"95": 91,
	"107": 105,
	// "110": 108,
	// "112": 110,
	// "114": 112,
	// "116": 114,
	// "118": 116,
	// "121": 118,
	// "123": 121,
	"124": 108,  // PARTIAL / MISSING
	"132": 130,  // PARTIAL
	"160": 158,
	"162": 160,
	"191": 189,
	"212": 208,
	"275": 271,
	"279": 277,  // PARTIAL
	"284": 282,
	"295": 288,
	"339": 337,
	"358": 356,
	"381": 376,
	"399": 397,
	"403": 401,
	"467": 465,
	"621": 617,
	"643": 637,
	"669": 667,  // PARTIAL
	"700": 698,
	"725": 723,  // PARTIAL
	"803": 773,
	"825": 823,
	"831": 829,  // PARTIAL
	"927": 925,  // PARTIAL
	"1204": 1202,
	"1300": 1298,
	"1325": 1323,
	"1328": 1325,
	"1356": 1350,
	"1361": 1359,
	"1364": 1362,
	"1381": 1379,
	"1386": 1383,
	"1394": 1391,
	"1419": 1416,
	"1423": 1419,
	"1432": 1430,
	"1434": 1432,
	"1446": 1439,
	"1448": 1446,
	"1451": 1448,
	"1464": 1451,
	"1479": 1464,
	"1501": 1479,
	"1505": 1503,
	"1531": 1529,
	"1557": 1550,
	"1560": 1558,
	"1564": 1560,
	"1575": 1564,
	"1578": 1575,
	"1585": 1578,
	"1594": 1585,
	"1598": 1594,
	"1641": 1639,
	"1667": 1665,  // PARTIAL
	"1730": 1720,
	"1739": 1730,
	"1750": 1745,
	"1781": 1751,
	"1787": 1785,
	"1789": 1787,
	"1791": 1789,
	"1800": 1792,
	"1802": 1800,
	"1819": 1817,
	"1927": 1925,
	"1994": 1992,
	"1997": 1995,
	"2021": 2019,
	"2031": 2029,
	"2114": 2112, // PARTIAL
	"2166": 2164,
	"2175": 2173,
	"2182": 2180,
	"2198": 2196,
	"2740": 2236,
	"2754": 2752,
	"2764": 2762,
	"2812": 2810,
	"2826": 2824,
	"2836": 2834,
	"2876": 2874,
	"2880": 2877,
	"2882": 2880,
	"2894": 2892,
	"2956": 2954,
	"2958": 2956,
	"3049": 3041,
	"3183": 3181,
	"3210": 3208,
	"3212": 3210,
	"3275": 3273,
	"3306": 3304,
	"3313": 3311,
	"3375": 3373,
	"3452": 3450,
	"3635": 3633,
	"3726": 3723,
	"3737": 3735,
	"3773": 3771,
	"3813": 3811,
	"3876": 3874,
	"3907": 3904,
	// END: 3994
}

#UNALIGNED_ENC_PATH:                "gs://zetta-research-nico/hippocampus/low_res_enc_c4"
// #RIGID_FIELD_PATH:                  "gs://zetta-research-nico/hippocampus/rigid/field"
#RIGID_ENC_PATH:                    "gs://zetta-research-nico/hippocampus/rigid_w_scale/low_res_enc_c4_rigid"
#AFFINE_TRANSFORM_PATH:             "gs://zetta-research-nico/hippocampus/affine_w_scale/transform_LMedS"
#AFFINE_FIELD_PATH:                 "gs://zetta-research-nico/hippocampus/pairwise/affine_w_scale/field"
#AFFINE_INV_FIELD_PATH:             "gs://zetta-research-nico/hippocampus/pairwise/affine_w_scale_inv/field"
#AFFINE_ENC_PATH:                   "gs://zetta-research-nico/hippocampus/pairwise/affine_w_scale_inv/low_res_enc_c4"

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
	[1301 * #REFERENCE_RES[2], 3994 * #REFERENCE_RES[2]],
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
local_test:      false
debug: false

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for z in list.Range(1300, 3994, 1) {
		// 			#SIFT_TEMPLATE & {
		// 				_z_offset: #Z_SKIP_MAP["\(z)"] - z
		// 				_bounds: [
		// 					[#ROI_BOUNDS[0][0], #ROI_BOUNDS[0][1]],
		// 					[#ROI_BOUNDS[1][0], #ROI_BOUNDS[1][1]],
		// 					[z * #REFERENCE_RES[2], (z+1) * #REFERENCE_RES[2]],
		// 				]
		// 				dst_resolution: [6144, 6144, 45]
		// 			}
		// 		},
		// 	]
		// },
		#GEN_AFFINE_TEMPLATE & {
			_bounds: #ROI_BOUNDS
			dst_resolution: [6144, 6144, 45]
			dst: path: #AFFINE_FIELD_PATH
		},
		#INVERT_FIELD_TEMPLATE & {
			_bounds: #ROI_BOUNDS
			dst_resolution: [6144, 6144, 45]
		},
		#WARP_ENC_TEMPLATE & {
			_bounds: #ROI_BOUNDS
			dst_resolution: [6144, 6144, 45]
		},
		#WARP_ENC_TEMPLATE & {
			_bounds: #ROI_BOUNDS
			dst_resolution: [6144*2, 6144*2, 45]
		},
		#WARP_ENC_TEMPLATE & {
			_bounds: #ROI_BOUNDS
			dst_resolution: [6144*4, 6144*4, 45]
		}
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
			ratio_test_fraction: 0.5
			ensure_scale_boundaries:  [-0.85, 1.2]
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
				{"@type": "torch.add", other: 128, "@mode": "partial"},
				{"@type": "to_uint8", "@mode": "partial"},
				{"@type": "lambda", "lambda_str": "lambda x: x[2:3,:,:,:]"}
			]
		}
		tgt: {
			"@type": "build_cv_layer"
			path:    #RIGID_ENC_PATH
			read_procs: [
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
		path: #AFFINE_TRANSFORM_PATH // + "/z\(_z_offset)"
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
			path:    #AFFINE_TRANSFORM_PATH
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
		path:                 _
		// info_field_overrides: #FIELD_INFO_OVERRIDE & {
		// 	_dst_resolution: dst_resolution
		// }
		// on_info_exists:       "overwrite"
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
			path:    #AFFINE_FIELD_PATH
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFFINE_INV_FIELD_PATH
		// info_reference_path: op_kwargs.src.path
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
			path:    #AFFINE_INV_FIELD_PATH
			data_resolution: [6144, 6144, 45]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFFINE_ENC_PATH
		// info_reference_path: op_kwargs.src.path
		// on_info_exists:      "overwrite"
	}
}
