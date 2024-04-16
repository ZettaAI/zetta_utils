import "math"

import "list"

#UNALIGNED_IMG_PATH:                "gs://dacey-human-retina-001-montaging/montage_prod_384croprender_128cropenc_final/img_warped/final"
#TARGET_IMG_PATH:                   "gs://dacey-human-retina-001-alignment-temp/rigid/img"
#TARGET_BINARIZED_DEFECT_MASK_PATH: "gs://dacey-human-retina-001-alignment-temp/rigid/defect_mask_refine"
#TARGET_ENC_PATH:                   "gs://dacey-human-retina-001-alignment-temp/rigid/enc"
#TARGET_MASKED_ENC_PATH:            "gs://dacey-human-retina-001-alignment-temp/rigid/enc_masked"
#ALIGNED_FIELD_PATH:                "gs://dacey-human-retina-001-alignment-temp/pairwise/field/fwd"
#ALIGNED_FIELD_INV_PATH:            "gs://dacey-human-retina-001-alignment-temp/pairwise/field/inv"
#ALIGNED_IMG_PATH:                  "gs://dacey-human-retina-001-alignment-temp/pairwise/img"
#ALIGNED_ENC_PATH:                  "gs://dacey-human-retina-001-alignment-temp/pairwise/enc"
#ALIGNED_MISD_PATH:                 "gs://dacey-human-retina-001-alignment-temp/pairwise/misd_raw"

#RIGID_FIELD_PATH:                  "gs://dacey-human-retina-001-alignment-temp/rigid/field"
#ACED_COARSE_FIELD_PATH:            "gs://dacey-human-retina-001-alignment-temp/aced/afield_try_2560nm_iter18000_rig3.0_lr0.0001"
#ACED_FIELD_PATH:                   "gs://dacey-human-retina-001-alignment-temp/aced/afield_try_640nm_iter9000_rig3.0_lr0.0001_final_v1"
#COMBINED_FIELD_PATH:               "gs://dacey-human-retina-001-alignment/field/v1"
#COMBINED_IMG_PATH:                 "gs://dacey-human-retina-001-alignment/img/v1"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]

#HIGH_RES: [40, 40, 50]

#DATASET_BOUNDS: [
	[0 * #HIGH_RES[0], 8192 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 8192 * #HIGH_RES[1]],
	[1 * #HIGH_RES[2], 3030 * #HIGH_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #HIGH_RES[0], 8192 * #HIGH_RES[0]],
	[0 * #HIGH_RES[1], 8192 * #HIGH_RES[1]],
	[1 * #HIGH_RES[2], 3030 * #HIGH_RES[2]],
]


_ALIGN_PARAMS: {
	"1280": {sm: 200, num_iter: 700, lr: 0.01},
	"640": {sm: 150, num_iter: 700, lr: 0.015},
	"320": {sm: 100, num_iter: 700, lr: 0.03},
	"160": {sm: 50, num_iter: 500, lr: 0.05},
	"80":  {sm: 50, num_iter: 300, lr: 0.1},
	"40":  {sm: 25, num_iter: 200, lr: 0.1},
}

#Z_OFFSETS: [-1,-2]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "dacey-human-retina-001"
worker_cluster_name:    "zutils"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240206"
// worker_resources: {
// 	"nvidia.com/gpu": "1"
// }
worker_resource_requests: {
	memory: "21000Mi"       // sized for n1-highmem-4
}
worker_replicas: 300
local_test:      false
// checkpoint_interval_sec: 60
// debug: true
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for i in list.Range(0, 6, 1) {
		// 			#ENCODE_UNALIGNED_TEMPLATE & {
		// 				_model: #ENCODER_MODELS[i]
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for i in list.Range(0, 6, 1) {
		// 			#MASK_ENCODINGS_TEMPLATE & {
		// 				dst_resolution: #ENCODER_MODELS[i].dst_resolution
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for z in #Z_OFFSETS {
		// 			#COMPUTE_FIELD_TEMPLATE & {
		// 				_z_offset: z
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for z in #Z_OFFSETS {
		// 			#INVERT_FIELD_TEMPLATE & {
		// 				_z_offset: z
		// 				dst_resolution: #HIGH_RES
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for z in #Z_OFFSETS {
		// 			#WARP_IMG_TEMPLATE & {
		// 				_z_offset: z
		// 				dst_resolution: #HIGH_RES
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for i in list.Range(1, 2, 1) for z in #Z_OFFSETS {
		// 			#ENCODE_ALIGNED_TEMPLATE & {
		// 				_model:    #ENCODER_MODELS_MISD[i]
		// 				_z_offset: z
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for z in #Z_OFFSETS {
		// 			#MISD_TEMPLATE & {
		// 				_z_offset: z
		// 			}
		// 		},
		// 	]
		// },

		// #WARP_AFIELD_TEMPLATE,
		// #WARP_FINAL_TEMPLATE,
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [20, 40, 80, 160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: #COMBINED_IMG_PATH
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: #ALIGNED_MISD_PATH + "/z-1"
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: #ALIGNED_MISD_PATH + "/z-2"
		// 			}
		// 		},
		// 	]
		// }
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [80, 160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: #TARGET_IMG_PATH
		// 			}
		// 		},
		// 	]
		// }
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: "precomputed://gs://dacey-human-retina-001-alignment-temp/rigid/tissue_mask"
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: "precomputed://gs://dacey-human-retina-001-alignment-temp/pairwise/misd_manual/z-1"
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: "precomputed://gs://dacey-human-retina-001-alignment-temp/rigid/defect_mask_refine"
		// 			}
		// 		},
		// 	]
		// }
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [80, 160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: #ALIGNED_FIELD_INV_PATH + "/z-1"
		// 				op_kwargs: src: interpolation_mode: "field"
		// 			}
		// 		},
		// 	]
		// },
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [80, 160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: #ALIGNED_FIELD_INV_PATH + "/z-2"
		// 				op_kwargs: src: interpolation_mode: "field"
		// 			}
		// 		},
		// 	]
		// }
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for res in [80, 160, 320, 640] {
		// 			#DOWNSAMPLE_FINAL_TEMPLATE & {
		// 				dst_resolution: [res, res, 50]
		// 				dst: path: "gs://dacey-human-retina-001-alignment-temp/aced/img_aligned_try_640nm_iter9000_rig3.0_lr0.0001_final_v1"
		// 				op_kwargs: src: interpolation_mode: "img"
		// 			}
		// 		},
		// 	]
		// }
		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		//for z in [100, 200, 300, 400, 500, 600, 701, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1802, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900] {
		// 		for z in [100] {
		// 			#UPSAMPLE_COARSE_FIELD_TEMPLATE & {
		// 				_z: z
		// 				dst_resolution: [640, 640, 50]
		// 				dst: path: #ACED_FIELD_PATH
		// 				op_kwargs: src: path: #ACED_COARSE_FIELD_PATH
		// 				op_kwargs: src: data_resolution: [2560, 2560, 50]
		// 			}
		// 		}
		// 	]
		// }

		// {
		// 	"@type": "mazepa.sequential_flow"
		// 	stages: [
		// 		for z in [100, 200, 300, 400, 500, 600, 701, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1802, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900] {
		// 		// for z in [2000] {
		// 			#WARP_RIGID_IMG_TEMPLATE & {
		// 				_z: z
		// 				dst_resolution: [40, 40, 50]
		// 				dst: path: "gs://dacey-human-retina-001-alignment-temp/aced/img_aligned_try_640nm_iter9000_rig3.0_lr0.0001_final_v1"
		// 				op_kwargs: src: path: #TARGET_IMG_PATH
		// 			}
		// 		}
		// 	]
		// }

		// #DOWNSAMPLE_FINAL_TEMPLATE & {
		// 	dst_resolution: [20, 20, 50]
		// 	op_kwargs: src: data_resolution: [10, 10, 50]
		// 	dst: path: "gs://dacey-human-retina-001-alignment/img/v1"
		// 	processing_chunk_sizes: [[2048, 2048, 16]]
		// },
		// #DOWNSAMPLE_FINAL_TEMPLATE & {
		// 	dst_resolution: [40, 40, 50]
		// 	op_kwargs: src: data_resolution: [20, 20, 50]
		// 	dst: path: "gs://dacey-human-retina-001-alignment/img/v1"
		// 	processing_chunk_sizes: [[1024, 1024, 16]]
		// },
		// #DOWNSAMPLE_FINAL_TEMPLATE & {
		// 	dst_resolution: [80, 80, 50]
		// 	op_kwargs: src: data_resolution: [40, 40, 50]
		// 	dst: path: "gs://dacey-human-retina-001-alignment/img/v1"
		// 	processing_chunk_sizes: [[512, 512, 16]]
		// }




		
	]
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ENCODER_MODELS: [
	{
		path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [1, 1, 1]
		dst_resolution: [40, 40, 50]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [2, 2, 1]
		dst_resolution: [80, 80, 50]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [4, 4, 1]
		dst_resolution: [160, 160, 50]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [8, 8, 1]
		dst_resolution: [320, 320, 50]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [16, 16, 1]
		dst_resolution: [640, 640, 50]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M3_M8_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		dst_resolution: [1280, 1280, 50]
	},
]

#ENCODER_MODELS_MISD: [
	{},
	{
		// path: "gs://alignment_models/general_encoders_2023/32_64_C1/2023-11-20.static-2.0.1-model.jit"
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [2, 2, 1]
		dst_resolution: [80, 80, 50]
	}
]

#MISD_MODELS: [
	{
		path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/4.0.1_dsfactor2_thr2.0_lr0.0001_z2/last.ckpt.model.spec.json"
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/aced_misd_general/4.0.1_dsfactor2_thr2.0_lr0.0001_z2/last.ckpt.model.spec.json" // not an error for dacey
	}
]

#FIELD_INFO_OVERRIDE: {
	_highest_resolution: _
	type:                "image"
	data_type:           "float32"
	num_channels:        2
	scales: [
		for i in list.Range(0, 6, 1) {
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
		for i in list.Range(0, 6, 1) {
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

#MASK_ENCODINGS_TEMPLATE: {
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src, mask: torch.where(mask > 0, 0, src)" // where(cond, true, false)
	}
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	dst_resolution:         _
	expand_bbox_resolution: true
	skip_intermediaries:    true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #TARGET_ENC_PATH
		}
		mask: {
			"@type": "build_cv_layer"
			path:    #TARGET_BINARIZED_DEFECT_MASK_PATH
			data_resolution: [320,320,50]
			interpolation_mode: "nearest"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #TARGET_MASKED_ENC_PATH
		info_reference_path: op_kwargs.src.path
	}
}

#ENCODE_UNALIGNED_TEMPLATE: {
	_model: {
		path: string
		res_change_mult: [int, int, int]
		dst_resolution: [int, int, int]
	}
	let max_cv_chunk_size = [
		list.Min([#DST_INFO_CHUNK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1] - #ROI_BOUNDS[0][0]) / dst_resolution[0])]),
		list.Min([#DST_INFO_CHUNK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1] - #ROI_BOUNDS[1][0]) / dst_resolution[1])])
	]
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/max_cv_chunk_size[0]/dst_resolution[0]) * max_cv_chunk_size[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/max_cv_chunk_size[1]/dst_resolution[1]) * max_cv_chunk_size[1]]),
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
				tile_size:       list.Min([1024, max_cv_chunk_size[0]])
				ds_factor:       op.res_change_mult[0]
				output_channels: 1
			}
		}
		crop_pad: [16, 16, 0]
		res_change_mult: _model.res_change_mult
	}
	dst_resolution: _model.dst_resolution
	processing_chunk_sizes: [max_chunk_size, [max_cv_chunk_size[0], max_cv_chunk_size[1], 1]]
	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0] - 2*#HIGH_RES[2]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	skip_intermediaries:    true
	expand_bbox_processing: true
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #TARGET_IMG_PATH
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #TARGET_ENC_PATH
		info_field_overrides: #ENC_INFO_OVERRIDE & {
			_highest_resolution: #HIGH_RES
		}
		on_info_exists: "overwrite"
	}
}

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
		#STAGE_TMPL & {  // 1280
			dst_resolution: [#HIGH_RES[0] * 32, #HIGH_RES[1] * 32, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["1280"]
		},
		#STAGE_TMPL & {  // 640
			dst_resolution: [#HIGH_RES[0] * 16, #HIGH_RES[1] * 16, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["640"]
		},
		#STAGE_TMPL & {  // 320
			dst_resolution: [#HIGH_RES[0] * 8, #HIGH_RES[1] * 8, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["320"]
		},
		#STAGE_TMPL & {  // 160
			dst_resolution: [#HIGH_RES[0] * 4, #HIGH_RES[1] * 4, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["160"]
		},
		#STAGE_TMPL & {  // 80
			dst_resolution: [#HIGH_RES[0] * 2, #HIGH_RES[1] * 2, #HIGH_RES[2]]
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["80"]
		},
		#STAGE_TMPL & {  // 40
			dst_resolution: #HIGH_RES
			fn: #STAGE_TMPL.fn & _ALIGN_PARAMS["40"]
		},
	]

	tgt_offset: [0, 0, _z_offset]
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
		path:    #ALIGNED_FIELD_PATH + "/z\(_z_offset)"
		info_reference_path: src.path
		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_highest_resolution: #HIGH_RES
		}
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: #ALIGNED_FIELD_PATH + "/tmp/z\(_z_offset)"
	tmp_layer_factory: {
		"@type":              "build_cv_layer"
		"@mode":              "partial"
		info_field_overrides: dst.info_field_overrides
		on_info_exists:       "overwrite"
	}
}

#INVERT_FIELD_TEMPLATE: {
	_z_offset: int

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
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
			path:    #ALIGNED_FIELD_PATH + "/z\(_z_offset)"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #ALIGNED_FIELD_INV_PATH + "/z\(_z_offset)"
		info_reference_path: op_kwargs.src.path
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
				offset: [0, 0, _z_offset]
				resolution: dst_resolution
			}]
		}
		field: {
			"@type": "build_cv_layer"
			path:    #ALIGNED_FIELD_INV_PATH + "/z\(_z_offset)"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #ALIGNED_IMG_PATH + "/z\(_z_offset)"
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
	let max_cv_chunk_size = [
		list.Min([#DST_INFO_CHUNK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1] - #ROI_BOUNDS[0][0]) / dst_resolution[0])]),
		list.Min([#DST_INFO_CHUNK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1] - #ROI_BOUNDS[1][0]) / dst_resolution[1])])
	]
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/max_cv_chunk_size[0]/dst_resolution[0]) * max_cv_chunk_size[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/max_cv_chunk_size[1]/dst_resolution[1]) * max_cv_chunk_size[1]]),
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
				tile_size:       list.Min([1024, max_cv_chunk_size[0]])
				ds_factor:       op.res_change_mult[0]
				output_channels: 1
			}
		}
		crop_pad: [16, 16, 0]
		res_change_mult: _model.res_change_mult
	}
	dst_resolution: _model.dst_resolution
	processing_chunk_sizes: [max_chunk_size, [max_cv_chunk_size[0], max_cv_chunk_size[1], 1]]
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
			path:    #ALIGNED_IMG_PATH + "/z\(_z_offset)"
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #ALIGNED_ENC_PATH + "/z\(_z_offset)"
		info_field_overrides: #ENC_INFO_OVERRIDE & {
			_highest_resolution: #HIGH_RES
		}
		on_info_exists: "overwrite"
	}
}

#MISD_TEMPLATE: {
	_z_offset: int

	let max_cv_chunk_size = [
		list.Min([#DST_INFO_CHUNK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1] - #ROI_BOUNDS[0][0]) / dst_resolution[0])]),
		list.Min([#DST_INFO_CHUNK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1] - #ROI_BOUNDS[1][0]) / dst_resolution[1])])
	]
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/max_cv_chunk_size[0]/dst_resolution[0]) * max_cv_chunk_size[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/max_cv_chunk_size[1]/dst_resolution[1]) * max_cv_chunk_size[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn:      {
			"@type":    "MisalignmentDetector"
			if math.Abs(_z_offset) == 1 {
				model_path: #MISD_MODELS[0].path
			}
			if math.Abs(_z_offset) == 2 {
				model_path: #MISD_MODELS[1].path
			}
			apply_sigmoid: true
		}
		crop_pad: [16, 16, 0]
	}
	dst_resolution: [80, 80, 50]
	processing_chunk_sizes: [max_chunk_size, [max_cv_chunk_size[0], max_cv_chunk_size[1], 1]]
	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
	}
	skip_intermediaries: true
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #ALIGNED_ENC_PATH + "/z\(_z_offset)"
		}
		tgt: {
			"@type": "build_cv_layer"
			path:    #TARGET_ENC_PATH
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #ALIGNED_MISD_PATH + "/z\(_z_offset)"
		info_reference_path:  op_kwargs.tgt.path
		info_field_overrides: {
			data_type: "uint8"
		}
		on_info_exists: "overwrite"
	}
}


#WARP_AFIELD_TEMPLATE: {
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "field"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: [40, 40, 50]
	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], 100*dst_resolution[2]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], 101*dst_resolution[2]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #RIGID_FIELD_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #ACED_FIELD_PATH
			data_resolution: [640, 640, 50]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COMBINED_FIELD_PATH
		info_reference_path: op_kwargs.src.path
	}
}

#WARP_FINAL_TEMPLATE: {
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
	dst_resolution: [10, 10, 50]
	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], 100*dst_resolution[2]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], 101*dst_resolution[2]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_IMG_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #COMBINED_FIELD_PATH
			data_resolution: [40, 40, 50]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #COMBINED_IMG_PATH
		info_reference_path: op_kwargs.src.path
	}
}

#DOWNSAMPLE_FINAL_TEMPLATE: {
	let max_cv_chunk_size = [
		list.Min([#DST_INFO_CHUNK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1] - #ROI_BOUNDS[0][0]) / dst_resolution[0])]),
		list.Min([#DST_INFO_CHUNK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1] - #ROI_BOUNDS[1][0]) / dst_resolution[1])])
	]
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/max_cv_chunk_size[0]/dst_resolution[0]) * max_cv_chunk_size[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/max_cv_chunk_size[1]/dst_resolution[1]) * max_cv_chunk_size[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "lambda"
		lambda_str: "lambda src: src"
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
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
			path:    dst.path
			data_resolution: [dst_resolution[0]/2, dst_resolution[1]/2, dst_resolution[2]]
			interpolation_mode: _ | *"img"
		}
	}
	dst: {
		"@type":     "build_cv_layer"
		path:        _
		info_add_scales:     [dst_resolution]
		info_add_scales_mode: "merge"
		info_reference_path: path
		on_info_exists:      "overwrite"
	}
}

#UPSAMPLE_COARSE_FIELD_TEMPLATE: {
	_z: int
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/#DST_INFO_CHUNK_SIZE[0]/dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/#DST_INFO_CHUNK_SIZE[1]/dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	dst_resolution:         _
	expand_bbox_resolution: true
	skip_intermediaries:    true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], _z * #HIGH_RES[2]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], (_z + 1) * #HIGH_RES[2]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    _
			data_resolution: _
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
	}
}


#WARP_RIGID_IMG_TEMPLATE: {
	_z: int
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
		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], _z * #HIGH_RES[2]]
		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], (_z + 1) * #HIGH_RES[2]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    _
		}
		field: {
			"@type": "build_cv_layer"
			path:    #ACED_FIELD_PATH
			data_resolution: [640, 640, 50]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
	}
}


#DOWNSAMPLE_FINAL_TEMPLATE: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "lambda"
		lambda_str: "lambda src: src"
	}
	dst_resolution: _
	processing_chunk_sizes: _
	processing_crop_pads: [[0, 0, 0]]
	skip_intermediaries:    true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [0, 512, 1]
		end_coord: [6144, 512+6144, 1+3029]
		resolution: [40,40,50]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    dst.path
			data_resolution: _
			interpolation_mode: "img"
		}
	}
	dst: {
		"@type":     "build_cv_layer"
		path:        _
		cv_kwargs: {
			compress: false
		}
		// info_add_scales:     [dst_resolution]
		// info_add_scales_mode: "merge"
		// info_reference_path: path
		// on_info_exists:      "overwrite"
	}
}