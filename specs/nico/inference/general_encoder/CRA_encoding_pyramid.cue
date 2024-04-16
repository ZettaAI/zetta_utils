import "math"

import "list"

// #SRC_PATH: "gs://dkronauer-ant-001-alignment/test-231114-103-50/pair-z50-103/imgs_warped/-2"           // M4
// #SRC_PATH: "gs://dkronauer-ant-001-alignment/test-231114-103-50/pair-z50-103-32nm/imgs_warped/-1"      // M3 SM10
#SRC_PATH: "gs://dkronauer-ant-001-alignment/test-231114-103-50/pair-z50-103-32nm-sm25/imgs_warped/-1"   // M3 SM25
// #SRC_PATH: "precomputed://gs://dkronauer-ant-001-raw/brain"

#DST_PATH: "gs://tmp_2w/nico/cra8/enc_20231208/z1_32_sm25"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#TASK_SIZE: [2048, 2048, 1]

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	// end_coord: [32768, 32768, 6112]
	end_coord: [12800, 12032, 6112]
	resolution: [32, 32, 42]
}

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 80]
	// end_coord: [32768, 32768, 3001]
	end_coord: [int, int, int] | *[12800, 12032, 90]
	resolution: [32, 32, 42]
}

#SCALES: [
	for i in list.Range(0, 5, 1) {
		let ds_factor = [math.Pow(2, i), math.Pow(2, i), 1]
		let vx_res = [ for j in [0, 1, 2] {#DATASET_BOUNDS.resolution[j] * ds_factor[j]}]
		let ds_offset = [ for j in [0, 1, 2] {
			__div(#DATASET_BOUNDS.start_coord[j], ds_factor[j])
		}]
		let ds_size = [ for j in [0, 1, 2] {
			__div((#DATASET_BOUNDS.end_coord[j] - ds_offset[j]), ds_factor[j])
		}]

		chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
		resolution:   vx_res
		encoding:     "raw"
		key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
		voxel_offset: ds_offset
		size:         ds_size
	},
]

#CHANNEL_COUNT: 1
#MODELS: [
	{
		path: "gs://alignment_models/general_encoders_2023/32_32_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [1, 1, 1]
	},
	{
		// path: "gs://alignment_models/general_encoders_2023/32_64_C1/2023-11-20.static-2.0.1-model.jit"
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/2.0.0_M3_M4_C1_lr2e-05_locality1.0_similarity0.0_l10.06-0.06_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [2, 2, 1]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_128_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [4, 4, 1]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_256_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [8, 8, 1]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_512_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [16, 16, 1]
	},
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/5_M3_M8_conv5_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/5_M4_M9_conv5_lr0.00002_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/7_M5_M10_conv5_lr0.0001_equi0.5_post1.06_fmt0.8_cns_all/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/6_M6_M11_conv5_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// },
	// {
	// 	path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/6_M7_M12_conv5_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
	// 	res_change_mult: [32, 32, 1]
	// },
]

#FLOW_TEMPLATE: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn:      {
			"@type":    "BaseEncoder"
			model_path: string
		} | {
			"@type":     "BaseCoarsener"
			model_path:  string
			tile_pad_in: int
			tile_size:   int
			ds_factor:   int
			output_channels: int
		}
		crop_pad: [16, 16, 0]
		res_change_mult: [int, int, int]
	}
	dst_resolution: [int, int, int]
	processing_chunk_sizes: [[int, int, int]]
	processing_crop_pads: [[0, 0, 0]]
	bbox: #ROI_BOUNDS
	skip_intermediaries: true
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}
	dst: {
		"@type":               "build_cv_layer"
		path:                  #DST_PATH
		info_field_overrides?: _
		on_info_exists:        "overwrite"
	}
}

"@type": "mazepa.execute"
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		// #FLOW_TEMPLATE & {
		// 	op: fn: {
		// 		"@type":    "BaseEncoder"
		// 		model_path: #MODELS[0].path
		// 	}
		// 	op: res_change_mult: #MODELS[0].res_change_mult
		// 	dst_resolution: #DATASET_BOUNDS.resolution
		// 	processing_chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#SCALES[0].chunk_sizes[0][j], #TASK_SIZE[j]])}]]
		// 	dst: info_field_overrides: {
		// 		type:         "image"
		// 		num_channels: #CHANNEL_COUNT
		// 		data_type:    "int8"
		// 		scales:       #SCALES
		// 	}
		// },
		for i in list.Range(1, 2, 1) {
			let res_mult = [math.Pow(2, i), math.Pow(2, i), 1]
			let dst_res = [ for j in [0, 1, 2] {#DATASET_BOUNDS.resolution[j] * res_mult[j]}]
			let trunc_tasksize = [ for j in [0, 1, 2] {list.Min([#SCALES[i].size[j], #TASK_SIZE[j]])}]
			let roi_pad = [ for j in [0, 1, 2] {
				__mod((trunc_tasksize[j] - (__mod(#SCALES[i].size[j], trunc_tasksize[j]))), trunc_tasksize[j])
			}]
			#FLOW_TEMPLATE & {
				op: fn: {
					"@type":     "BaseCoarsener"
					model_path:  #MODELS[i].path
					tile_pad_in: #FLOW_TEMPLATE.op.crop_pad[0] * #MODELS[i].res_change_mult[0]
					tile_size:   1024
					ds_factor:   #MODELS[i].res_change_mult[0]
					output_channels: #CHANNEL_COUNT
				}
				op: res_change_mult: #MODELS[i].res_change_mult
				dst_resolution: dst_res
				processing_chunk_sizes: [trunc_tasksize]
				dst: info_field_overrides: {
					type:         "image"
					num_channels: #CHANNEL_COUNT
					data_type:    "int8"
					scales:       #SCALES
				}
				bbox: #ROI_BOUNDS & {
					end_coord: [ for j in [0, 1, 2] {#ROI_BOUNDS.end_coord[j] + roi_pad[j]*math.Pow(2, i)}]
				}
			}
		},
	]
}
