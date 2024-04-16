import "math"

import "list"

#UNALIGNED_IMG_PATH: "https://td.princeton.edu/sseung-test1/ca3-alignment-temp/full_section_imap4"
#UNALIGNED_ENC_PATH: "gs://zetta-research-nico/hippocampus/low_res_enc_c4"

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
	[1790 * #REFERENCE_RES[2], 1791 * #REFERENCE_RES[2]],
]

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240224"
worker_resources: {
    "nvidia.com/gpu": "1"
}
// worker_resource_requests: {
//     memory: "10000Mi"
// }
worker_replicas: 20
local_test:      true
debug: true

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		#ENCODE_UNALIGNED_TEMPLATE & {
			_model: #ENCODER_MODELS[0]
		},
		#ENCODE_UNALIGNED_TEMPLATE & {
			_model: #ENCODER_MODELS[1]
		},
		#ENCODE_UNALIGNED_TEMPLATE & {
			_model: #ENCODER_MODELS[2]
		}
	]
}


#ENCODER_MODELS: [
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.1_M6_M11_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		dst_resolution: [6144, 6144, #REFERENCE_RES[2]]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M7_M12_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		dst_resolution: [12288, 12288, #REFERENCE_RES[2]]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M7_M12_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		dst_resolution: [24576, 24576, #REFERENCE_RES[2]]
	},
]

#ENC_INFO_OVERRIDE: {
	type:                "image"
	data_type:           "int8"
	num_channels:        4
	scales: [
		for i in list.Range(11, 14, 1) {
			let res_factor = [math.Pow(2, i), math.Pow(2, i), 1]
			let vx_res = [ for j in [0, 1, 2] {#REFERENCE_RES[j] * res_factor[j]}]
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


#ENCODE_UNALIGNED_TEMPLATE: {
	_model: {
		path: string
		res_change_mult: [int, int, int]
		dst_resolution: [int, int, int]
	}
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((#ROI_BOUNDS[0][1]-#ROI_BOUNDS[0][0])/dst_resolution[0])]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((#ROI_BOUNDS[1][1]-#ROI_BOUNDS[1][0])/dst_resolution[1])]),
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
				output_channels: 4
			}
		}
		crop_pad: [16, 16, 0]
		res_change_mult: _model.res_change_mult
	}
	dst_resolution: _model.dst_resolution
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
			path:    #UNALIGNED_IMG_PATH
			data_resolution: [48, 48, #REFERENCE_RES[2]] //data_resolution: [384, 384, #REFERENCE_RES[2]]
			interpolation_mode: "img"
			cv_kwargs: {"cache": true}
		}
	}
	dst: {
		"@type":              "build_cv_layer"
		path:                 #UNALIGNED_ENC_PATH
        // info_field_overrides: #ENC_INFO_OVERRIDE
		// on_info_exists: "overwrite"
	}
}

