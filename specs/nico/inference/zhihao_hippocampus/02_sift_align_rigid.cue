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
	"110": 108,
	"112": 110,
	"114": 112,
	"116": 114,
	"118": 116,
	"121": 118,
	"123": 121,
	"160": 158,
	"162": 160,
	"191": 189,
	"212": 208,
	"275": 271,
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
	"700": 698,
	"803": 773,
	"825": 823,
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
	"1997": 1995,
	"2021": 2019,
	"2031": 2029,
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

#UNALIGNED_IMG_PATH: "https://td.princeton.edu/sseung-test1/ca3-alignment-temp/full_section_imap4"
#UNALIGNED_ENC_PATH: "gs://zetta-research-nico/hippocampus/low_res_enc_m6_1536nm"
#RIGID_TRANSFORM_PATH: "gs://zetta-research-nico/hippocampus/rigid_transforms_3x3_LMedS"

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
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240229"
// worker_resources: {
//     "nvidia.com/gpu": "1"
// }
worker_resource_requests: {
    memory: "10000Mi"
}
worker_replicas: 20
local_test:      true
debug: true

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		// {
        //     #ENCODE_UNALIGNED_TEMPLATE & {
        //         _model: #ENCODER_MODELS[0]
        //     }
		// },
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for z in list.Range(926, 927, 1) {
					#SIFT_TEMPLATE & {
						_z_offset: #Z_SKIP_MAP["\(z)"] - z
						_bounds: [
							[#ROI_BOUNDS[0][0], #ROI_BOUNDS[0][1]],
							[#ROI_BOUNDS[1][0], #ROI_BOUNDS[1][1]],
							[z * #REFERENCE_RES[2], (z+1) * #REFERENCE_RES[2]],
						]
						dst_resolution: [1536, 1536, 45]
					}
				},
			]
		}
	]
}


#ENCODER_MODELS: [
	// {},  // [5, 5, 50]
	// {},  // [10, 10, 50]
	// {},  // [20, 20, 50]
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [1, 1, 1]
	// 	dst_resolution: [384, 384, #REFERENCE_RES[2]]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [2, 2, 1]
	// 	dst_resolution: [384, 384, #REFERENCE_RES[2]]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [4, 4, 1]
	// 	dst_resolution: [768, 768, #REFERENCE_RES[2]]
	// },
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [8, 8, 1]
		dst_resolution: [1536, 1536, #REFERENCE_RES[2]]
	},
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [16, 16, 1]
	// 	dst_resolution: [640, 640, #REFERENCE_RES[2]]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M3_M8_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// 	dst_resolution: [1280, 1280, #REFERENCE_RES[2]]
	// },

	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.1_M6_M11_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// 	dst_resolution: [6144, 6144, #REFERENCE_RES[2]]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M7_M12_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// 	dst_resolution: [12288, 12288, #REFERENCE_RES[2]]
	// },
	
]

// #ENC_INFO_OVERRIDE: {
// 	type:                "image"
// 	data_type:           "int8"
// 	num_channels:        1
// 	scales: [
// 		for i in list.Range(6, 9, 1) {
// 			let res_factor = [math.Pow(2, i), math.Pow(2, i), 1]
// 			let vx_res = [ for j in [0, 1, 2] {#REFERENCE_RES[j] * res_factor[j]}]
// 			let ds_offset = [ for j in [0, 1, 2] {
// 				#DATASET_BOUNDS[j][0] / vx_res[j]// technically should be floor, but it's 0 anyway
// 			}]
// 			let ds_size = [ for j in [0, 1, 2] {
// 				math.Ceil((#DATASET_BOUNDS[j][1] - #DATASET_BOUNDS[j][0]) / vx_res[j])
// 			}]

// 			chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
// 			resolution:   vx_res
// 			encoding:     "raw"
// 			key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
// 			voxel_offset: ds_offset
// 			size:         ds_size
// 		},
// 	]
// }


#ENCODE_UNALIGNED_TEMPLATE: {
	_model: {
		path: string
		res_change_mult: [int, int, int]
		dst_resolution: [int, int, int]
	}
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			model_path: _model.path
			if _model.res_change_mult[0] == 1 {
				"@type": "BaseEncoder"
			}
			if _model.res_change_mult[0] > 1 {
				"@type":         "BaseCoarsener"
				tile_pad_in:     op.crop_pad[0] * op.res_change_mult[0]
				tile_size:       512
				ds_factor:       op.res_change_mult[0]
				output_channels: 1
			}
		}
		crop_pad: [16, 16, 0]
		res_change_mult: _model.res_change_mult
	}
	dst_resolution: _model.dst_resolution
	processing_chunk_sizes: [max_chunk_size, [1024, 1024, 1]]
	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	skip_intermediaries:    true
	expand_bbox_processing: true
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_IMG_PATH
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #UNALIGNED_ENC_PATH
        info_field_overrides: {
            "type": "image",
            "data_type": "int8",
            "num_channels": 1,
            "scales": [
                {
                    chunk_sizes: [[1024, 1024, 1]],
                    resolution: _model.dst_resolution,
                    encoding: "raw",
                    key: "\(dst_resolution[0])_\(dst_resolution[1])_\(dst_resolution[2])",
                    voxel_offset: [
                        math.Floor(#DATASET_BOUNDS[0][0]/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0],
                        math.Floor(#DATASET_BOUNDS[1][0]/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1],
                        #DATASET_BOUNDS[2][0] / dst_resolution[2]
                    ],
                    size: [
                        math.Ceil(#DATASET_BOUNDS[0][1]/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0] - voxel_offset[0],
                        math.Ceil(#DATASET_BOUNDS[1][1]/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1] - voxel_offset[1],
                        (#DATASET_BOUNDS[2][1]-#DATASET_BOUNDS[2][0]) / dst_resolution[2]
                    ]
                }
            ]

        }
		on_info_exists: "overwrite"
	}
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
			transformation_mode: "rigid"
			estimate_mode: "lmeds"
			ratio_test_fraction: 0.5
			ensure_scale_boundaries:  [0.91, 1.15]
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
				// {"@type": "lambda", "lambda_str": "lambda x: x[1:2,:,:,:]"}
			]
		}
		tgt: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_ENC_PATH
			read_procs: [
				{"@type": "torch.add", other: 128, "@mode": "partial"},
				{"@type": "to_uint8", "@mode": "partial"},
				// {"@type": "lambda", "lambda_str": "lambda x: x[1:2,:,:,:]"}
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
		path: #RIGID_TRANSFORM_PATH + "/z\(_z_offset)"
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