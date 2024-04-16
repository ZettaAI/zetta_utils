import "math"

import "list"

#TARGET_IMG_PATH:                   "gs://dkronauer-ant-001-raw/brain"
#TARGET_ENC_PATH:                   "gs://tmp_2w/nico/cra8/enc/z0_cns"
#TARGET_MASKED_ENC_PATH:            "gs://tmp_2w/nico/cra8/enc/z0_masked_cns"
#TARGET_DEFECT_MASK_PATH:           "gs://tmp_2w/nico/cra8/defect_mask"
#TARGET_BINARIZED_DEFECT_MASK_PATH: "gs://tmp_2w/nico/cra8/binarized_defect_mask"
#TARGET_RESIN_MASK_PATH:            "gs://tmp_2w/nico/cra8/resin_mask"
#MISALIGNED_FIELD_PATH:             "gs://tmp_2w/nico/cra8/bad_field_fwd_align_cns" //_M6xSM\(_ALIGN_PARAMS["256"].sm)_M5xSM\(_ALIGN_PARAMS["128"].sm)_M4xSM\(_ALIGN_PARAMS["64"].sm)"
#MISALIGNED_IMG_PATH:               "gs://tmp_2w/nico/cra8/bad_img_fwd_align_cns" //_M6xSM\(_ALIGN_PARAMS["256"].sm)_M5xSM\(_ALIGN_PARAMS["128"].sm)_M4xSM\(_ALIGN_PARAMS["64"].sm)"
#MISALIGNED_ENC_PATH:               "gs://tmp_2w/nico/cra8/bad_enc_fwd_align_cns" //_M6xSM\(_ALIGN_PARAMS["256"].sm)_M5xSM\(_ALIGN_PARAMS["128"].sm)_M4xSM\(_ALIGN_PARAMS["64"].sm)"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]
#HIGH_RES: [32, 32, 42]

#DATASET_BOUNDS: [
	[0 * #HIGH_RES[0], 12800 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 12032 * #HIGH_RES[1]],
	[0 * #HIGH_RES[2], 6112 * #HIGH_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #HIGH_RES[0], 12800 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 12032 * #HIGH_RES[1]],
	[4000 * #HIGH_RES[2], 4020 * #HIGH_RES[2]],
]

_ALIGN_PARAMS: {
	"512": {sm: 150, num_iter: 700, lr: 0.015},
	"256": {sm: 100, num_iter: 700, lr: 0.03},
	"128": {sm: 75, num_iter: 500, lr: 0.05},
	"64":  {sm: 50, num_iter: 300, lr: 0.1},
	"32":  {sm: 25, num_iter: 200, lr: 0.1},
}

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20231213"
worker_resources: {
	"nvidia.com/gpu": "1"
}
worker_replicas: 100
local_test:      false
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		{
		// 			"@type": "mazepa.sequential_flow"
		// 			stages: [
		// 				#DEFECT_DETECTION_TEMPLATE,
		// 				#DEFECT_POSTPROCESS_TEMPLATE,
		// 			]
		// 		},
		// 		#RESIN_DETECTION_TEMPLATE,
		// 	]
		// },
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for i in list.Range(0, 5, 1) {
		// 			#ENCODE_UNALIGNED_TEMPLATE & {
		// 				_model: #ENCODER_MODELS[i]
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for i in list.Range(1, 4, 1) {
		// 			#DOWNSAMPLE_MASK_TEMPLATE & {
		// 				src_resolution: #ENCODER_MODELS[i].dst_resolution
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for i in list.Range(0, 5, 1) {
		// 			#MASK_ENCODINGS_TEMPLATE & {
		// 				dst_resolution: #ENCODER_MODELS[i].dst_resolution
		// 				if i == 3 || i == 4 {
		// 					_width: 5
		// 				}
		// 				if i == 1 || i == 2 {
		// 					_width: 3
		// 				}
		// 				if i == 0 {
		// 					_width: 1
		// 				}
		// 			}
		// 		},
		// 	]
		// },
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for i in [-1] {
					#COMPUTE_FIELD_TEMPLATE & {
						_z_offset: i
					}
				},
			]
		},
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for i in [1, 2] {
					#WARP_IMG_TEMPLATE & {
						_z_offset:      i
						dst_resolution: #HIGH_RES
					}
				},
			]
		},
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for i in list.Range(1, 5, 1) for j in [1, 2] {
					#ENCODE_ALIGNED_TEMPLATE & {
						_model:    #ENCODER_MODELS[i]
						_z_offset: j
					}
				},
			]
		},
	]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ENCODER_MODELS: [
	{
		path: "gs://alignment_models/general_encoders_2023/32_32_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [1, 1, 1]
		dst_resolution: [32, 32, 42]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_64_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [2, 2, 1]
		dst_resolution: [64, 64, 42]
	},
	{
		// path: "gs://alignment_models/general_encoders_2023/32_128_C1/2023-11-20.static-2.0.1-model.jit"
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M5_conv2_unet2_lr0.0001_equi0.5_post1.4_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
		res_change_mult: [4, 4, 1]
		dst_resolution: [128, 128, 42]
	},
	{
		// path: "gs://alignment_models/general_encoders_2023/32_256_C1/2023-11-20.static-2.0.1-model.jit"
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/4_M3_M6_conv3_unet1_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
		res_change_mult: [8, 8, 1]
		dst_resolution: [256, 256, 42]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_512_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [16, 16, 1]
		dst_resolution: [512, 512, 42]
	},
]

#FIELD_INFO_OVERRIDE: {
	_highest_resolution: _
	type:                "image"
	data_type:           "float32"
	num_channels:        2
	scales: [
		for i in list.Range(0, 5, 1) {
			let res_factor = [math.Pow(2, i), math.Pow(2, i), 1]
			let vx_res = [ for j in [0, 1, 2] {_highest_resolution[j] * res_factor[j]}]
			let ds_offset = [ for j in [0, 1, 2] {
				#DATASET_BOUNDS[j][0] / vx_res[j]// technically should be floor, but it's 0 anyway
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((#DATASET_BOUNDS[j][1] - #DATASET_BOUNDS[j][0]) / vx_res[j])
			}]

			chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
			resolution: vx_res
			encoding:   "zfpc"
			zfpc_correlated_dims: [true, true, false, false]
			zfpc_tolerance: 0.001953125
			key:            "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
			voxel_offset:   ds_offset
			size:           ds_size
		},
	]
}

#ENC_INFO_OVERRIDE: {
	_highest_resolution: _
	type:                "image"
	data_type:           "int8"
	num_channels:        1
	scales: [
		for i in list.Range(0, 5, 1) {
			let res_factor = [math.Pow(2, i), math.Pow(2, i), 1]
			let vx_res = [ for j in [0, 1, 2] {_highest_resolution[j] * res_factor[j]}]
			let ds_offset = [ for j in [0, 1, 2] {
				#DATASET_BOUNDS[j][0] / vx_res[j]// technically should be floor, but it's 0 anyway
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((#DATASET_BOUNDS[j][1] - #DATASET_BOUNDS[j][0]) / vx_res[j])
			}]

			chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
			resolution:   vx_res
			encoding:     "raw"
			key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
			voxel_offset: ds_offset
			size:         ds_size
		},
	]
}

// #DEFECT_CHUNK_SIZE: [2048, 2048, 1]

// #DEFECT_CROP_PAD: [256, 256, 0] // tested with tile=512
// #DEFECT_DETECTION_TEMPLATE: {
// 	let max_chunk_size = [
// 		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
// 		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
// 		1,
// 	]

// 	"@type": "build_subchunkable_apply_flow"
// 	op: {
// 		"@type": "VolumetricCallableOperation"
// 		fn: {
// 			"@type":     "DefectDetector"
// 			model_path:  "gs://zetta_lee_fly_cns_001_models/jit/20221114-defects-step50000.static-1.11.0.jit"
// 			tile_pad_in: op.crop_pad[0]
// 			tile_size:   #DEFECT_CHUNK_SIZE[0]
// 		}
// 		crop_pad: #DEFECT_CROP_PAD
// 	}
// 	dst_resolution: [64, 64, 42]
// 	processing_chunk_sizes: [max_chunk_size, #DEFECT_CHUNK_SIZE]
// 	processing_crop_pads: [[0, 0, 0], #DEFECT_CROP_PAD]
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}
// 	skip_intermediaries:    true
// 	expand_bbox_processing: true
// 	op_kwargs: {
// 		src: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_IMG_PATH
// 		}
// 	}
// 	dst: {
// 		"@type": "build_cv_layer"
// 		path:    #TARGET_DEFECT_MASK_PATH
// 		info_add_scales_ref: {
// 			chunk_sizes: [#DEFECT_CHUNK_SIZE]
// 			encoding:   "raw"
// 			resolution: #HIGH_RES
// 			size: [
// 				math.Ceil((#DATASET_BOUNDS[0][1] - #DATASET_BOUNDS[0][0]) / #HIGH_RES[0]),
// 				math.Ceil((#DATASET_BOUNDS[1][1] - #DATASET_BOUNDS[1][0]) / #HIGH_RES[1]),
// 				math.Ceil((#DATASET_BOUNDS[2][1] - #DATASET_BOUNDS[2][0]) / #HIGH_RES[2]),
// 			]
// 			voxel_offset: [0, 0, 0]
// 		}
// 		info_add_scales: [dst_resolution]
// 		info_field_overrides: {
// 			data_type:    "uint8"
// 			num_channels: 1
// 			type:         "image"
// 		}
// 		on_info_exists: "overwrite"
// 	}
// }

// #DEFECT_POSTPROCESS_CROP_PAD: [32, 32, 0]
// #DEFECT_POSTPROCESS_TEMPLATE: {
// 	let max_chunk_size = [
// 		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
// 		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
// 		1,
// 	]

// 	"@type": "build_subchunkable_apply_flow"
// 	fn: {
// 		"@type": "lambda"
// 		// set masked area to 0
// 		lambda_str: "lambda src, mask: torch.where(mask > 0, 0, src)" // where(cond, true, false)
// 	}
// 	dst_resolution: [64, 64, 42]
// 	processing_chunk_sizes: [max_chunk_size, #DEFECT_CHUNK_SIZE]
// 	processing_crop_pads: [[0, 0, 0], #DEFECT_POSTPROCESS_CROP_PAD]
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}
// 	skip_intermediaries:    true
// 	expand_bbox_processing: true
// 	op_kwargs: {
// 		src: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_DEFECT_MASK_PATH
// 		}
// 		mask: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_DEFECT_MASK_PATH
// 			read_procs: [
// 				{
// 					"@type": "compare", "@mode": "partial"
// 					mode:    ">="
// 					value:   100
// 				},
// 				{
// 					"@type": "to_uint8", "@mode": "partial"
// 				},
// 				{
// 					// remove thin line from mask
// 					"@type": "kornia_opening", "@mode": "partial"
// 					width:   11
// 				},
// 				{
// 					// grow mask a little
// 					"@type": "kornia_dilation", "@mode": "partial"
// 					width:   3
// 				},
// 			]
// 		}
// 	}
// 	dst: {
// 		"@type":             "build_cv_layer"
// 		path:                #TARGET_BINARIZED_DEFECT_MASK_PATH
// 		info_reference_path: #TARGET_DEFECT_MASK_PATH
// 		on_info_exists:      "overwrite"
// 		write_procs: [
// 			{
// 				"@type": "compare", "@mode": "partial"
// 				mode:    ">="
// 				value:   100
// 			},
// 			{
// 				// remove small islands that are likely FPs
// 				"@type": "filter_cc", "@mode": "partial"
// 				mode:    "keep_large"
// 				thr:     320
// 			},
// 			{
// 				// connect disconnected folds
// 				"@type": "kornia_closing", "@mode": "partial"
// 				width:   25
// 			},
// 			{
// 				"@type": "to_uint8", "@mode": "partial"
// 			},
// 		]
// 	}
// }

// #DOWNSAMPLE_MASK_TEMPLATE: {
// 	let max_chunk_size = [
// 		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
// 		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
// 		1,
// 	]

// 	"@type": "build_interpolate_flow"
// 	mode:    "mask"
// 	src_resolution: [number, number, number]
// 	dst_resolution: [src_resolution[0] * 2, src_resolution[1] * 2, src_resolution[2]]
// 	chunk_size: max_chunk_size
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}

// 	src: {
// 		"@type": "build_cv_layer"
// 		path:    #TARGET_BINARIZED_DEFECT_MASK_PATH
// 	}
// 	dst: {
// 		"@type":             "build_cv_layer"
// 		path:                #TARGET_BINARIZED_DEFECT_MASK_PATH
// 		on_info_exists:      "overwrite"
// 		info_reference_path: #TARGET_ENC_PATH
// 		info_field_overrides: {
// 			data_type: "uint8"
// 		}
// 	}
// }

// #MASK_ENCODINGS_TEMPLATE: {
// 	let max_chunk_size = [
// 		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
// 		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
// 		1,
// 	]

// 	"@type": "build_subchunkable_apply_flow"
// 	fn: {
// 		"@type":    "lambda"
// 		lambda_str: "lambda src, mask: torch.where(mask > 0, 0, src)" // where(cond, true, false)
// 	}
// 	processing_chunk_sizes: [max_chunk_size]
// 	processing_crop_pads: [[0, 0, 0]]
// 	dst_resolution:         _
// 	expand_bbox_resolution: true
// 	skip_intermediaries:    true
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}
// 	op_kwargs: {
// 		src: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_ENC_PATH
// 		}
// 		mask: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_BINARIZED_DEFECT_MASK_PATH
// 			data_resolution: [
// 				list.Max([64, dst_resolution[0]]),
// 				list.Max([64, dst_resolution[1]]),
// 				dst_resolution[2],
// 			]
// 			interpolation_mode: "nearest"
// 			read_procs: [
// 				{
// 					"@type": "kornia_dilation"
// 					"@mode": "partial"
// 					width:   3
// 				},
// 			]
// 		}
// 	}
// 	dst: {
// 		"@type":             "build_cv_layer"
// 		path:                #TARGET_MASKED_ENC_PATH
// 		info_reference_path: op_kwargs.src.path
// 	}
// }

// #RESIN_CHUNK_SIZE: [2048, 2048, 1]
// #RESIN_CROP_PAD: [32, 32, 0]
// #RESIN_DETECTION_TEMPLATE: {
// 	let max_chunk_size = [
// 		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
// 		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
// 		1,
// 	]

// 	"@type": "build_subchunkable_apply_flow"
// 	op: {
// 		"@type": "VolumetricCallableOperation"
// 		fn: {
// 			"@type":                 "ResinDetector"
// 			model_path:              "gs://zetta_lee_fly_cns_001_models/jit/20221115-resin-step29000.static-1.11.0.jit"
// 			tile_pad_in:             op.crop_pad[0]
// 			tile_size:               #RESIN_CHUNK_SIZE[0]
// 			tissue_filter_threshold: 0
// 		}
// 		crop_pad: #RESIN_CROP_PAD
// 	}
// 	dst_resolution: [256, 256, 42]
// 	processing_chunk_sizes: [max_chunk_size, #RESIN_CHUNK_SIZE]
// 	processing_crop_pads: [[0, 0, 0], #RESIN_CROP_PAD]
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}
// 	skip_intermediaries:    true
// 	expand_bbox_processing: true
// 	op_kwargs: {
// 		src: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_IMG_PATH
// 		}
// 	}
// 	dst: {
// 		"@type": "build_cv_layer"
// 		path:    #TARGET_RESIN_MASK_PATH
// 		info_add_scales_ref: {
// 			chunk_sizes: [#RESIN_CHUNK_SIZE]
// 			encoding:   "raw"
// 			resolution: #HIGH_RES
// 			size: [
// 				math.Ceil((#DATASET_BOUNDS[0][1] - #DATASET_BOUNDS[0][0]) / #HIGH_RES[0]),
// 				math.Ceil((#DATASET_BOUNDS[1][1] - #DATASET_BOUNDS[1][0]) / #HIGH_RES[1]),
// 				math.Ceil((#DATASET_BOUNDS[2][1] - #DATASET_BOUNDS[2][0]) / #HIGH_RES[2]),
// 			]
// 			voxel_offset: [0, 0, 0]
// 		}
// 		info_add_scales: [dst_resolution]
// 		info_field_overrides: {
// 			data_type:    "uint8"
// 			num_channels: 1
// 			type:         "image"
// 		}
// 		on_info_exists: "overwrite"
// 	}
// }

// #ENCODE_UNALIGNED_TEMPLATE: {
// 	_model: {
// 		path: string
// 		res_change_mult: [int, int, int]
// 		dst_resolution: [int, int, int]
// 	}
// 	let max_chunk_size = [
// 		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
// 		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
// 		1,
// 	]

// 	"@type": "build_subchunkable_apply_flow"
// 	op: {
// 		"@type": "VolumetricCallableOperation"
// 		fn: {
// 			model_path: _model.path
// 			if _model.res_change_mult[0] == 1 {
// 				"@type": "BaseEncoder"
// 			}
// 			if _model.res_change_mult[0] > 1 {
// 				"@type":         "BaseCoarsener"
// 				tile_pad_in:     op.crop_pad[0] * op.res_change_mult[0]
// 				tile_size:       1024
// 				ds_factor:       op.res_change_mult[0]
// 				output_channels: 1
// 			}
// 		}
// 		crop_pad: [16, 16, 0]
// 		res_change_mult: _model.res_change_mult
// 	}
// 	dst_resolution: _model.dst_resolution
// 	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
// 	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}
// 	skip_intermediaries:    true
// 	expand_bbox_processing: true
// 	op_kwargs: {
// 		src: {
// 			"@type": "build_cv_layer"
// 			path:    #TARGET_IMG_PATH
// 		}
// 	}
// 	dst: {
// 		"@type":              "build_cv_layer"
// 		path:                 #TARGET_ENC_PATH
// 		info_field_overrides: #ENC_INFO_OVERRIDE & {
// 			_highest_resolution: #HIGH_RES
// 		}
// 		on_info_exists: "overwrite"
// 	}
// }

#STAGE_TMPL: {
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	expand_bbox_processing: true
	expand_bbox_resolution: true
	fn: {
		"@type":  "align_with_online_finetuner"
		"@mode":  "partial"
		sm:       int
		num_iter: int
		lr:       float
	}
}

#COMPUTE_FIELD_TEMPLATE: {
	_z_offset: int

	"@type": "build_compute_field_multistage_flow"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	stages: [
		#STAGE_TMPL & {  // 512
			dst_resolution: [#HIGH_RES[0] * 16, #HIGH_RES[1] * 16, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["512"]
		},
		#STAGE_TMPL & {  // 256
			dst_resolution: [#HIGH_RES[0] * 8, #HIGH_RES[1] * 8, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["256"]
		},
		#STAGE_TMPL & {  // 128
			dst_resolution: [#HIGH_RES[0] * 4, #HIGH_RES[1] * 4, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["128"]
		},
		#STAGE_TMPL & {  // 64
			dst_resolution: [#HIGH_RES[0] * 2, #HIGH_RES[1] * 2, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["64"]
		},
		#STAGE_TMPL & {  // 32
			dst_resolution: #HIGH_RES
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["32"]
		},
	]

	src_offset: [0, 0, -_z_offset]
	offset_resolution: #HIGH_RES

	src: {
		"@type": "build_cv_layer"
		path:    #TARGET_MASKED_ENC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #TARGET_MASKED_ENC_PATH
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #MISALIGNED_FIELD_PATH + "/z\(_z_offset)"

		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_highest_resolution: #HIGH_RES
		}
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: #MISALIGNED_FIELD_PATH + "/tmp/z\(_z_offset)"
	tmp_layer_factory: {
		"@type":              "build_cv_layer"
		"@mode":              "partial"
		info_field_overrides: dst.info_field_overrides
		on_info_exists:       "overwrite"
	}
}

#WARP_IMG_TEMPLATE: {
	_z_offset: int

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #TARGET_IMG_PATH
			index_procs: [{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, -_z_offset]
				resolution: dst_resolution
			}]
		}
		field: {
			"@type": "build_cv_layer"
			path:    #MISALIGNED_FIELD_PATH + "/z\(_z_offset)"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #MISALIGNED_IMG_PATH + "/z\(_z_offset)"
		info_reference_path: op_kwargs.src.path
	}
}

#ENCODE_ALIGNED_TEMPLATE: {
	_model: {
		path: string
		res_change_mult: [int, int, int]
		dst_resolution: [int, int, int]
	}
	_z_offset: int
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
				tile_size:       1024
				ds_factor:       op.res_change_mult[0]
				output_channels: 1
			}
		}
		crop_pad: [16, 16, 0]
		res_change_mult: _model.res_change_mult
	}
	dst_resolution: _model.dst_resolution
	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
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
			path:    #MISALIGNED_IMG_PATH + "/z\(_z_offset)"
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #MISALIGNED_ENC_PATH + "/z\(_z_offset)"
		info_field_overrides: #ENC_INFO_OVERRIDE & {
			_highest_resolution: #HIGH_RES
		}
		on_info_exists: "overwrite"
	}
}
