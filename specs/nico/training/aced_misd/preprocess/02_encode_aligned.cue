import "math"
import "list"

#BASE_PATH: "gs://zetta-research-nico/encoder/"
#TGT_IMG_PATH: #BASE_PATH + "datasets/" // + k
#WARPED_SRC_IMG_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/warped_img"
#DST_TGT_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/tgt_enc_2023"
#DST_WARPED_SRC_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/warped_enc_2023"

#DATASETS: {
	"microns_pinky": {
		"contiguous": true
		"bounds": [[0, 262144], [0, 131072], [0, 10240]]
		"resolution": [32, 32, 40]
	}
	"microns_basil": {
		"contiguous": true
		"bounds": [[0, 819200], [0, 983040], [0, 400]]
		"resolution": [32, 32, 40]
	},
	"microns_minnie": {
		"contiguous": false
		"bounds": [[0, 1703936], [0, 1441792], [0, 320]]
		"resolution": [32, 32, 40]
	},
	"microns_interneuron": {
		"contiguous": false
		"bounds": [[0, 720896], [0, 720896], [0, 1280]]
		"resolution": [32, 32, 40]
	},
	"aibs_v1dd": {
		"contiguous": false
		"bounds": [[0.0, 1231667.2], [0.0, 834355.2], [0.0, 1080.0]]
		"resolution": [38.8, 38.8, 45.0]
	},
	"kim_n2da": {
		"contiguous": true
		"bounds": [[0, 32768], [0, 32768], [0, 31050]]
		"resolution": [32, 32, 50]
	},
	"kim_pfc2022": {
		"contiguous": true
		"bounds": [[0, 229376], [0, 196608], [0, 7320]]
		"resolution": [32, 32, 40]
	},
	"kronauer_cra9": {
		"contiguous": true
		"bounds": [[0, 393216], [0, 327680], [0, 588]]
		"resolution": [32, 32, 42]
	},
	"kubota_001": {
		"contiguous": true
		"bounds": [[0, 204800], [0, 204800], [0, 12000]]
		"resolution": [40, 40, 40]
	},
	"lee_fanc": {
		"contiguous": false
		"bounds": [[0.0, 352256.0], [0.0, 951091.2], [0.0, 2700.0]]
		"resolution": [34.4, 34.4, 45.0]
	},
	"lee_banc": {
		"contiguous": false
		"bounds": [[0, 819200], [0, 1015808], [0, 900]]
		"resolution": [32, 32, 45]
	},
	"lee_ppc": {
		"contiguous": true
		"bounds": [[0, 98304], [0, 98304], [0, 36400]]
		"resolution": [32, 32, 40]
	},
	"lee_mosquito": {
		"contiguous": false
		"bounds": [[0, 704512], [0, 450560], [0, 2240]]
		"resolution": [32, 32, 40]
	},
	"lichtman_zebrafish": {
		"contiguous": false
		"bounds": [[0, 294912], [0, 393216], [0, 4560]]
		"resolution": [32, 32, 30]
	},
	"prieto_godino_larva": {
		"contiguous": true
		"bounds": [[0, 134976], [0, 144992], [0, 14400]]
		"resolution": [32, 32, 32]
	},
	"fafb_v15": {
		"contiguous": false
		"bounds": [[0, 884736], [0, 393216], [0, 2000]]
		"resolution": [32, 32, 40]
	},
	"lichtman_h01": {
		"contiguous": false
		"bounds": [[0, 3440640], [0, 1933312], [0, 198]]
		"resolution": [32, 32, 33]
	},
	"janelia_hemibrain": {
		"contiguous": true
		"bounds": [[0, 317824], [0, 331168], [0, 3296]]
		"resolution": [32, 32, 32]
	},
	"janelia_manc": {
		"contiguous": false
		"bounds": [[0, 262144], [0, 360448], [0, 5952]]
		"resolution": [32, 32, 32]
	},
	"nguyen_thomas_2022": {
		"contiguous": true
		"bounds": [[0, 998400], [0, 921600], [0, 400]]
		"resolution": [32, 32, 40]
	},
	"mulcahy_2022_16h": {
		"contiguous": true
		"bounds": [[0, 243712], [0, 73728], [0, 14700]]
		"resolution": [32, 32, 30]
	},
	"wildenberg_2021_vta_dat12a": {
		"contiguous": true
		"bounds": [[0, 82080], [0, 85184], [0, 7640]]
		"resolution": [32, 32, 40]
	},
	"bumbarber_2013": {
		"contiguous": true
		"bounds": [[0.0, 63897.6], [0.0, 63897.6], [0.0, 102400.0]]
		"resolution": [31.2, 31.2, 50.0]
	},
	"wilson_2019_p3": {
		"contiguous": true
		"bounds": [[0, 163840], [0, 229376], [0, 7020]]
		"resolution": [32, 32, 30]
	},
	"ishibashi_2021_em1": {
		"contiguous": true
		"bounds": [[0, 24576], [0, 16384], [0, 4544]]
		"resolution": [32, 32, 32]
	},
	"ishibashi_2021_em2": {
		"contiguous": true
		"bounds": [[0, 26624], [0, 18432], [0, 5376]]
		"resolution": [32, 32, 32]
	},
	"templier_2019_wafer1": {
		"contiguous": true
		"bounds": [[0, 294912], [0, 229376], [0, 6500]]
		"resolution": [32, 32, 50]
	},
	"templier_2019_wafer3": {
		"contiguous": true
		"bounds": [[0, 229376], [0, 196608], [0, 9750]]
		"resolution": [32, 32, 50]
	},
	"lichtman_octopus2022": {
		"contiguous": true
		"bounds": [[0, 229376], [0, 360448], [0, 3180]]
		"resolution": [32, 32, 30]
	}
}


#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]
#MAX_TASK_SIZE: [8192, 8192, 1]

#ENC_INFO_OVERRIDE: {
	_dataset_bounds: _
	_highest_resolution: _
	type:         "image"
	data_type:    "int8"
	num_channels: 1
	scales: [
		for i in list.Range(0, 4, 1) {
			let res_factor = [math.Pow(2, i), math.Pow(2, i), 1]
			let vx_res = [ for j in [0, 1, 2] {_highest_resolution[j] * res_factor[j]}]
			let ds_offset = [ for j in [0, 1, 2] {
				_dataset_bounds[j][0] / vx_res[j]  // technically should be floor, but it's
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((_dataset_bounds[j][1] - _dataset_bounds[j][0]) / vx_res[j])
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

#MODELS: [
	{
		path: "gs://alignment_models/general_encoders_2023/32_32_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [1, 1, 1]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_64_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [2, 2, 1]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_128_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [4, 4, 1]
	},
	{
		path: "gs://alignment_models/general_encoders_2023/32_256_C1/2023-11-20.static-2.0.1-model.jit"
		res_change_mult: [8, 8, 1]
	}
]

#ENCODE_TEMPLATE: {
	_bounds: _
	_high_resolution: [number, number, number]
	_layer_name: _
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1] - _bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1] - _bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
	]

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		operation_name: _layer_name
		fn: {
			"@type":     "BaseEncoder"
			model_path:  string
		} | {
			"@type":     "BaseCoarsener"
			model_path:  string
			tile_pad_in: int
			tile_size:   int
			ds_factor:   int
			output_channels: 1
		}
		crop_pad: [16, 16, 0]
		res_change_mult: [int, int, int]
	}
	dst_resolution: [number, number, number]
	processing_chunk_sizes: [max_chunk_size, [1024, 1024, 1]]
	processing_crop_pads: [[0, 0, 0], [16,16,0]]
	expand_bbox_resolution: true
	skip_intermediaries: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    _
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _
		info_field_overrides: #ENC_INFO_OVERRIDE & {
			_dataset_bounds: _bounds
			_highest_resolution: _high_resolution
		}
		on_info_exists: "overwrite"
	}
}



"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
worker_resources: {
	"nvidia.com/gpu":     1
}
worker_replicas:      200
batch_gap_sleep_sec:  0.1
do_dryrun_estimation: true
local_test:           false
worker_cluster_project: "zetta-research"
worker_cluster_region: "us-east1"
worker_cluster_name: "zutils-x3"
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for img_source in ["tgt", "warped_src"] {
			"@type": "mazepa.concurrent_flow"
			stages: [
				for key, dataset in #DATASETS {
					"@type": "mazepa.concurrent_flow"
					stages: [
						for i in list.Range(0, 4, 1) {
							#ENCODE_TEMPLATE & {
								_bounds: dataset.bounds,
								_high_resolution: dataset.resolution
								_layer_name: "Model \(i)"
								let _ds_factor = #MODELS[i].res_change_mult
								let res = [ for j in [0, 1, 2] {dataset.resolution[j] * _ds_factor[j]} ]
								if i == 0 {
									op: fn: "@type": "BaseEncoder"
								}
								if i > 0 {
									op: fn: {
										"@type":     "BaseCoarsener"
										tile_pad_in: #ENCODE_TEMPLATE.op.crop_pad[0] * _ds_factor[0]
										tile_size:   1024
										ds_factor:   _ds_factor[0]
									}
								}
								op: fn: model_path: #MODELS[i].path
								op: res_change_mult: _ds_factor
								dst_resolution: res
								if img_source == "tgt" {
									op_kwargs: src: path: #TGT_IMG_PATH + key
									dst: path: #DST_TGT_ENC_PATH + key + "/tgt_enc_2023"
								}
								if img_source == "warped_src" {
									op_kwargs: src: path: #WARPED_SRC_IMG_PATH + key + "/warped_img"
									dst: path: #DST_WARPED_SRC_ENC_PATH + key + "/warped_enc_2023"
								}
							}
						}
					]
				}
			]
		}
	]
}