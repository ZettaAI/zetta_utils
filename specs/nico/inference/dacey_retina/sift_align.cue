import "math"

import "list"

#UNALIGNED_IMG_PATH: "gs://dacey-human-retina-001-montaging/montage_prod_384croprender_128cropenc_final/img_warped/final"
#UNALIGNED_ENC_PATH: "gs://dacey-human-retina-001-alignment-temp/unaligned/enc"
#RIGID_TRANSFORM_PATH: "gs://tmp_2w/dacey-human-retina-001-alignment-temp/rigid_transforms_3x3_LMedS"

#KEYPOINT_DB: "nkem/sift_keypoints"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#MAX_TASK_SIZE: [8192, 8192, 1]
#REFERENCE_RES: [5, 5, 50]

#DATASET_BOUNDS: [
	[0 * #REFERENCE_RES[0], 65536 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 65536 * #REFERENCE_RES[1]],
	[1 * #REFERENCE_RES[2], 3030 * #REFERENCE_RES[2]],
]

#ROI_BOUNDS: [
	[0 * #REFERENCE_RES[0], 65536 * #REFERENCE_RES[0]],
	[0 * #REFERENCE_RES[1], 65536 * #REFERENCE_RES[1]],
	[1 * #REFERENCE_RES[2], 3030 * #REFERENCE_RES[2]],
]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "dacey-human-retina-001"
worker_cluster_name:    "zutils"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240120"
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
		// 	"@type": "mazepa.concurrent_flow"
		// 	stages: [
		// 		for i in list.Range(8, 9, 1) {
		// 			#ENCODE_UNALIGNED_TEMPLATE & {
		// 				_model: #ENCODER_MODELS[i]
		// 			}
		// 		},
		// 	]
		// },
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				for z in [-1, -2] {
					#SIFT_TEMPLATE & {
						_z_offset: z
						dst_resolution: [640, 640, 50]
					}
				},
			]
		}
	]
}


// #ENCODER_MODELS: [
// 	{},  // [5, 5, 50]
// 	{},  // [10, 10, 50]
// 	{},  // [20, 20, 50]
// 	{
// 		path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
// 		res_change_mult: [1, 1, 1]
// 		dst_resolution: [40, 40, 50]
// 	},
// 	{
// 		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
// 		res_change_mult: [2, 2, 1]
// 		dst_resolution: [80, 80, 50]
// 	},
// 	{
// 		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
// 		res_change_mult: [4, 4, 1]
// 		dst_resolution: [160, 160, 50]
// 	},
// 	{
// 		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
// 		res_change_mult: [8, 8, 1]
// 		dst_resolution: [320, 320, 50]
// 	},
// 	{
// 		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
// 		res_change_mult: [16, 16, 1]
// 		dst_resolution: [640, 640, 50]
// 	},
// 	{
// 		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M3_M8_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
// 		res_change_mult: [32, 32, 1]
// 		dst_resolution: [1280, 1280, 50]
// 	},
// ]

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
// 				tile_size:       512
// 				ds_factor:       op.res_change_mult[0]
// 				output_channels: 1
// 			}
// 		}
// 		crop_pad: [16, 16, 0]
// 		res_change_mult: _model.res_change_mult
// 	}
// 	dst_resolution: _model.dst_resolution
// 	processing_chunk_sizes: [max_chunk_size, [1024, 1024, 1]]
// 	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
// 	bbox: {
// 		"@type": "BBox3D.from_coords"
// 		start_coord: [#ROI_BOUNDS[0][0], #ROI_BOUNDS[1][0], #ROI_BOUNDS[2][0]]
// 		end_coord: [#ROI_BOUNDS[0][1], #ROI_BOUNDS[1][1], #ROI_BOUNDS[2][1]]
// 	}
// 	skip_intermediaries:    true
// 	expand_bbox_processing: true
// 	op_kwargs: {
// 		src: {
// 			"@type": "build_cv_layer"
// 			path:    #UNALIGNED_IMG_PATH
// 		}
// 	}
// 	dst: {
// 		"@type":              "build_cv_layer"
// 		path:                 #UNALIGNED_ENC_PATH
// 		info_field_overrides: #ENC_INFO_OVERRIDE
// 		on_info_exists: "overwrite"
// 	}
// }

// SIFT
#SIFT_TEMPLATE: {
	_z_offset: int

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/dst_resolution[1])]),
		1,
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":  "RigidTransform2D"
			ratio_test_fraction: 0.7
		}
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
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
			path:    #UNALIGNED_ENC_PATH
			read_procs: [
				{"@type": "torch.add", other: 128, "@mode": "partial"},
				{"@type": "to_uint8", "@mode": "partial"}
			]
		}
		tgt: {
			"@type": "build_cv_layer"
			path:    #UNALIGNED_ENC_PATH
			read_procs: [
				{"@type": "torch.add", other: 128, "@mode": "partial"},
				{"@type": "to_uint8", "@mode": "partial"}
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

