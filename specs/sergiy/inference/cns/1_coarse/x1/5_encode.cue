import "math"

import "list"

#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1"
#SRC_PATH:    "\(#BASE_FOLDER)/raw_img"
#DST_PATH:    "\(#BASE_FOLDER)/encodings"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1] // Will automatically get truncated if dataset becomes too small
#TASK_SIZE: [2048, 2048, 1]

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 6150]
	// end_coord: [32768, 32768, 3001]
	end_coord: [int, int, int] | *[32768, 36864, 6170]
	resolution: [32, 32, 45]
}

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	// end_coord: [32768, 32768, 7010]
	end_coord: [32768, 36864, 8000]
	resolution: [32, 32, 45]
}
#SCALES: [
	for i in list.Range(0, 10, 1) {
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

#MODELS: [
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [1, 1, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/3_M3_M4_conv1_unet3_lr0.0001_equi0.5_post1.6_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [2, 2, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/3_M3_M5_conv2_unet2_lr0.0001_equi0.5_post1.4_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [4, 4, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/4_M3_M6_conv3_unet1_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [8, 8, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/5_M3_M7_conv4_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/epoch=0-step=1584-backup.ckpt.model.spec.json"
		res_change_mult: [16, 16, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/5_M3_M8_conv5_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/5_M4_M9_conv5_lr0.00002_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/7_M5_M10_conv5_lr0.0001_equi0.5_post1.06_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/6_M6_M11_conv5_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-lee-fly-vnc-001-nico/training_artifacts/base_coarsener_cns/6_M7_M12_conv5_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
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
		}
		crop_pad: [16, 16, 0]
		res_change_mult: [int, int, int]
	}
	expand_bbox_processing: true
	dst_resolution: [int, int, int]
	processing_chunk_sizes: _
	processing_crop_pads: [0, 0, 0]
	bbox: #ROI_BOUNDS
	src: {
		"@type": "build_ts_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":               "build_cv_layer"
		path:                  #DST_PATH
		info_field_overrides?: _
		on_info_exists:        "overwite"
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x139"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:        20
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
batch_gap_sleep_sec:    1.0
local_test:             false

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#FLOW_TEMPLATE & {
			op: fn: {
				"@type":    "BaseEncoder"
				model_path: #MODELS[0].path
			}
			op: res_change_mult: #MODELS[0].res_change_mult
			dst_resolution: #DATASET_BOUNDS.resolution
			processing_chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#SCALES[0].chunk_sizes[0][j], #TASK_SIZE[j]])}]]
			dst: info_field_overrides: {
				type:         "image"
				num_channels: 1
				data_type:    "int8"
				scales:       #SCALES
			}
		},
		for i in list.Range(1, 10, 1) {
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
				}
				op: res_change_mult: #MODELS[i].res_change_mult
				dst_resolution: dst_res
				processing_chunk_sizes: [trunc_tasksize]
				bbox: #ROI_BOUNDS & {
					end_coord: [ for j in [0, 1, 2] {#ROI_BOUNDS.end_coord[j] + roi_pad[j]*math.Pow(2, i)}]
				}
			}
		},
	]
}
