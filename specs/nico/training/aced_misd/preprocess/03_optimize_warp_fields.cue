import "math"
import "list"

#BASE_PATH: "gs://zetta-research-nico/encoder/"
// #TGT_IMG_PATH: #BASE_PATH + "datasets/" // + k
#ORIGINAL_WARPED_SRC_IMG_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/warped_img"
#TGT_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/tgt_enc_2023"
#WARPED_SRC_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/warped_enc_2023"
#PERLIN_FIELD_PATH: #BASE_PATH + "misd/misalignment_fields/" // + k + "/raw_perlin"
#DST_FIELD_PATH: #BASE_PATH + "misd/misalignment_fields/" // + k + "/optimized_perlin" | "/no_perlin" + "/z\(_z_offset)"

#DST_WARPED_SRC_IMG_PATH: #BASE_PATH + "misd/img/" // + k + "/good_alignment" | "/bad_alignment" + "/z\(_z_offset)"
#DST_WARPED_SRC_ENC_PATH: #BASE_PATH + "misd/enc/" // + k + "/good_alignment" | "/bad_alignment" + "/z\(_z_offset)"


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
	// // "microns_minnie": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 1703936], [0, 1441792], [0, 320]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "microns_interneuron": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 720896], [0, 720896], [0, 1280]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "aibs_v1dd": {
	// // 	"contiguous": false
	// // 	"bounds": [[0.0, 1231667.2], [0.0, 834355.2], [0.0, 1080.0]]
	// // 	"resolution": [38.8, 38.8, 45.0]
	// // },
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
	// // "lee_fanc": {
	// // 	"contiguous": false
	// // 	"bounds": [[0.0, 352256.0], [0.0, 951091.2], [0.0, 2700.0]]
	// // 	"resolution": [34.4, 34.4, 45.0]
	// // },
	// // "lee_banc": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 819200], [0, 1015808], [0, 900]]
	// // 	"resolution": [32, 32, 45]
	// // },
	"lee_ppc": {
		"contiguous": true
		"bounds": [[0, 98304], [0, 98304], [0, 36400]]
		"resolution": [32, 32, 40]
	},
	// // "lee_mosquito": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 704512], [0, 450560], [0, 2240]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "lichtman_zebrafish": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 294912], [0, 393216], [0, 4560]]
	// // 	"resolution": [32, 32, 30]
	// // },
	"prieto_godino_larva": {
		"contiguous": true
		"bounds": [[0, 134976], [0, 144992], [0, 14400]]
		"resolution": [32, 32, 32]
	},
	// // "fafb_v15": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 884736], [0, 393216], [0, 2000]]
	// // 	"resolution": [32, 32, 40]
	// // },
	// // "lichtman_h01": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 3440640], [0, 1933312], [0, 198]]
	// // 	"resolution": [32, 32, 33]
	// // },
	"janelia_hemibrain": {
		"contiguous": true
		"bounds": [[0, 317824], [0, 331168], [0, 3296]]
		"resolution": [32, 32, 32]
	},
	// // "janelia_manc": {
	// // 	"contiguous": false
	// // 	"bounds": [[0, 262144], [0, 360448], [0, 5952]]
	// // 	"resolution": [32, 32, 32]
	// // },
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


#DST_INFO_CHUNK_SIZE: [2048, 2048, 1]
#MAX_TASK_SIZE: [8192, 8192, 1]
#PERLIN_FIELD_DS_FACTOR: math.Pow(2, 3)

#STAGE_TMPL: {
	_stage_bounds: _
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_stage_bounds[0][1] - _stage_bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_stage_bounds[1][1] - _stage_bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
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


#FIELD_INFO_OVERRIDE: {
	_dataset_bounds: _
	_highest_resolution: _
	type: "image"
	data_type: "float32",
	num_channels: 2,
	scales: [
		for i in list.Range(0, 3, 1) {
			let res_factor = [math.Pow(2, i), math.Pow(2, i), 1]
			let vx_res = [ for j in [0, 1, 2] {_highest_resolution[j] * res_factor[j]}]
			let ds_offset = [ for j in [0, 1, 2] {
				_dataset_bounds[j][0] / vx_res[j]  // technically should be floor, but it's 0 anyway
			}]
			let ds_size = [ for j in [0, 1, 2] {
				math.Ceil((_dataset_bounds[j][1] - _dataset_bounds[j][0]) / vx_res[j])
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
	]
}

#COMPUTE_FIELD_TEMPLATE: {
	_bounds: _
	_dst_resolution: [number, number, number]
	_layer_name: _
	_z_offset: int
	_use_perlin_field: *false | true

	"@type":     "build_compute_field_multistage_flow"
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	stages: [
		#STAGE_TMPL & {
			_stage_bounds: _bounds
			dst_resolution: [_dst_resolution[0] * 4, _dst_resolution[1] * 4, _dst_resolution[2]]
			fn: {
				sm:       25
				num_iter: 500
				lr:       0.05
			}
		},
		#STAGE_TMPL & {
			_stage_bounds: _bounds
			dst_resolution: [_dst_resolution[0] * 2, _dst_resolution[1] * 2, _dst_resolution[2]]
			fn: {
				sm:       25
				num_iter: 300
				lr:       0.1
			}
		},
		#STAGE_TMPL & {
			_stage_bounds: _bounds
			dst_resolution: _dst_resolution
			fn: {
				sm:       25
				num_iter: 200
				lr:       0.1
			}
		},
	]

	if _z_offset == 2 {
		src_offset: [0, 0, 1]  // src is already offset by 1
		offset_resolution: _dst_resolution
	}
	src: {
		"@type": "build_cv_layer"
		path:    #WARPED_SRC_ENC_PATH + _layer_name + "/warped_enc_2023"
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #TGT_ENC_PATH + _layer_name + "/tgt_enc_2023"
	}
	dst: {
		"@type":  "build_cv_layer"
		if _use_perlin_field == true {
			path: #PERLIN_FIELD_PATH + _layer_name + "/optimized_perlin/z\(_z_offset)"
		}
		if _use_perlin_field == false {
			path: #PERLIN_FIELD_PATH + _layer_name + "/no_perlin/z\(_z_offset)"
		}
		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_dataset_bounds: _bounds
			_highest_resolution: _dst_resolution
		}
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: dst.path + "/tmp"
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_field_overrides: #FIELD_INFO_OVERRIDE & {
			_dataset_bounds: _bounds
			_highest_resolution: _dst_resolution
		}
		on_info_exists: "overwrite"
	}
	if _use_perlin_field {
		src_field: {
			let ds_factor = [#PERLIN_FIELD_DS_FACTOR, #PERLIN_FIELD_DS_FACTOR, 1]
			"@type": "build_cv_layer"
			path:    #PERLIN_FIELD_PATH + _layer_name + "/raw_perlin"
			data_resolution: [ for j in [0, 1, 2] {_dst_resolution[j] * ds_factor[j]} ]
			interpolation_mode: "field"
		}
	}
}


#WARP_IMG_TEMPLATE: {
	_bounds: _
	_layer_name: _
	_z_offset: int
	_use_perlin_field: *false | true

	_src_field_path: _
	_dst_img_path: _
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1] - _bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1] - _bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
	]
	if _use_perlin_field == true {
		_src_field_path: #DST_FIELD_PATH + _layer_name + "/optimized_perlin/z\(_z_offset)"
		_dst_img_path: #DST_WARPED_SRC_IMG_PATH + _layer_name + "/bad_alignment/z\(_z_offset)"
	}
	if _use_perlin_field == false {
		_src_field_path: #DST_FIELD_PATH + _layer_name + "/no_perlin/z\(_z_offset)"
		_dst_img_path: #DST_WARPED_SRC_IMG_PATH + _layer_name + "/good_alignment/z\(_z_offset)"
	}

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: _
	processing_chunk_sizes: [max_chunk_size, [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	skip_intermediaries: true
	expand_bbox_processing: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path: #ORIGINAL_WARPED_SRC_IMG_PATH + _layer_name + "/warped_img"
			index_procs: [{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, _z_offset - 1]  // src is already offset by 1
				resolution: dst_resolution
			}]
		}
		field: {
			"@type": "build_cv_layer"
			path: _src_field_path
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path: _dst_img_path
		info_reference_path: op_kwargs.src.path
	}
}

#DOWNSAMPLE_FIELD_TEMPLATE: {
	_bounds: _
	_layer_name: _
	_z_offset: int
	_use_perlin_field: *false | true
	
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1] - _bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1] - _bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
	]

	"@type": "build_interpolate_flow"
	mode: "field"
	src_resolution: [number, number, number]
	dst_resolution: [src_resolution[0] * 2, src_resolution[1] * 2, src_resolution[2]]
	chunk_size: max_chunk_size
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}

	_path: _
	if _use_perlin_field == true {
		_path: #DST_FIELD_PATH + _layer_name + "/optimized_perlin/z\(_z_offset)"
	}
	if _use_perlin_field == false {
		_path: #DST_FIELD_PATH + _layer_name + "/no_perlin/z\(_z_offset)"
	}
	src: {
		"@type": "build_cv_layer"
		path: _path
	}
	dst: {
		"@type": "build_cv_layer"
		path: _path
	}

}

#FIELD_DIFF_TEMPLATE: {
	_bounds: _
	_layer_name: _
	_z_offset: int
	
	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1] - _bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1] - _bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
	]

	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "torch.sub", "@mode": "partial"
	}
	processing_chunk_sizes: [max_chunk_size]
	processing_crop_pads: [[0, 0, 0]]
	dst_resolution: _
	expand_bbox_resolution: true
	skip_intermediaries: true
	bbox: {
		"@type": "BBox3D.from_coords",
		start_coord: [_bounds[0][0], _bounds[1][0], _bounds[2][0]]
		end_coord: [_bounds[0][1], _bounds[1][1], _bounds[2][1]]
	}
	op_kwargs: {
		input: {
			"@type": "build_cv_layer"
			path: #DST_FIELD_PATH + _layer_name + "/optimized_perlin/z\(_z_offset)"
		}
		other: {
			"@type": "build_cv_layer"
			path: #DST_FIELD_PATH + _layer_name + "/no_perlin/z\(_z_offset)"
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path: #DST_FIELD_PATH + _layer_name + "/displacements/z\(_z_offset)"
		info_reference_path: #TGT_ENC_PATH + _layer_name + "/tgt_enc_2023"
		info_field_overrides: {
			data_type: "uint8"
		}
		on_info_exists: "overwrite"
		write_procs: [
			{
				"@type":    "lambda"
				lambda_str: "lambda data: (data.norm(dim=0, keepdim=True)*10.0).round().clamp(0, 255).byte()"
			}
		]
	}
}


#ENCODE_IMG_TEMPLATE: {
	_bounds: _
	_high_resolution: [number, number, number]
	_layer_name: _
	_z_offset: int
	_use_perlin_field: *false | true
	_model: {
		path: _
		res_change_mult: [int, int, int]
	}

	let max_chunk_size = [
		list.Min([#MAX_TASK_SIZE[0], math.Ceil((_bounds[0][1] - _bounds[0][0]) / #DST_INFO_CHUNK_SIZE[0] / dst_resolution[0]) * #DST_INFO_CHUNK_SIZE[0]]),
		list.Min([#MAX_TASK_SIZE[1], math.Ceil((_bounds[1][1] - _bounds[1][0]) / #DST_INFO_CHUNK_SIZE[1] / dst_resolution[1]) * #DST_INFO_CHUNK_SIZE[1]]),
		1
	]

	_src_img_path: _
	_dst_enc_path: _
	if _use_perlin_field == true {
		_src_img_path: #DST_WARPED_SRC_IMG_PATH + _layer_name + "/bad_alignment/z\(_z_offset)"
		_dst_enc_path: #DST_WARPED_SRC_ENC_PATH + _layer_name + "/bad_alignment/z\(_z_offset)"
	}
	if _use_perlin_field == false {
		_src_img_path: #DST_WARPED_SRC_IMG_PATH + _layer_name + "/good_alignment/z\(_z_offset)"
		_dst_enc_path: #DST_WARPED_SRC_ENC_PATH + _layer_name + "/good_alignment/z\(_z_offset)"
	}

	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		operation_name: _layer_name
		fn: {
			if _model.res_change_mult[0] == 1 {
				"@type":     "BaseEncoder"
			}
			if _model.res_change_mult[1] > 1 {
				"@type":     "BaseCoarsener"
				tile_pad_in: op.crop_pad[0]
				tile_size:   1024
				ds_factor:   _model.res_change_mult[0]
			}
			model_path:  _model.path
		}
		crop_pad: [16, 16, 0]
		res_change_mult: _model.res_change_mult
	}
	dst_resolution: [ for j in [0, 1, 2] {_high_resolution[j] * _model.res_change_mult[j]} ]
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
			path: _src_img_path
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path: _dst_enc_path
		info_reference_path: #TGT_ENC_PATH + _layer_name + "/tgt_enc_2023"
	}
}


#COMPUTE_FIELD_STAGE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
	worker_resources: {
		"nvidia.com/gpu":     "1"
	}
	worker_replicas:      300
	batch_gap_sleep_sec:  0.1
	do_dryrun_estimation: true
	local_test:           false
	worker_cluster_project: "zetta-research"
	worker_cluster_region: "us-east1"
	worker_cluster_name: "zutils-x3"
	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for key, dataset in #DATASETS {
				"@type": "mazepa.concurrent_flow"
				stages: [
					#COMPUTE_FIELD_TEMPLATE & {
						_bounds: dataset.bounds,
						_dst_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
					},
					#COMPUTE_FIELD_TEMPLATE & {
						_bounds: dataset.bounds,
						_dst_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
						_use_perlin_field: true
					},
					if dataset.contiguous {
						#COMPUTE_FIELD_TEMPLATE & {
							_bounds: dataset.bounds,
							_dst_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
						}
					},
					if dataset.contiguous {
						#COMPUTE_FIELD_TEMPLATE & {
							_bounds: dataset.bounds,
							_dst_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
							_use_perlin_field: true
						},
					}
				]
			}
		]
	}
}


#WARP_IMAGE_STAGE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
	worker_resources: {
		"memory":     "8Gi"
	}
	worker_replicas:      100
	batch_gap_sleep_sec:  0.1
	do_dryrun_estimation: true
	local_test:           false
	worker_cluster_project: "zetta-research"
	worker_cluster_region: "us-east1"
	worker_cluster_name: "zutils-x3"
	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for key, dataset in #DATASETS {
				"@type": "mazepa.concurrent_flow"
				stages: [
					#WARP_IMG_TEMPLATE & {
						_bounds: dataset.bounds,
						dst_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
					},
					#WARP_IMG_TEMPLATE & {
						_bounds: dataset.bounds,
						dst_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
						_use_perlin_field: true
					},
					if dataset.contiguous {
						#WARP_IMG_TEMPLATE & {
							_bounds: dataset.bounds,
							dst_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
						}
					}
					if dataset.contiguous {
						#WARP_IMG_TEMPLATE & {
							_bounds: dataset.bounds,
							dst_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
							_use_perlin_field: true
						},
					}
				]
			}
		]
	}
}

#DOWNSAMPLE_FIELD_STAGE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
	worker_resources: {
		"memory":     "8Gi"
	}
	worker_replicas:      100
	batch_gap_sleep_sec:  0.1
	do_dryrun_estimation: true
	local_test:           false
	worker_cluster_project: "zetta-research"
	worker_cluster_region: "us-east1"
	worker_cluster_name: "zutils-x3"
	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for key, dataset in #DATASETS {
				"@type": "mazepa.concurrent_flow"
				stages: [
					#DOWNSAMPLE_FIELD_TEMPLATE & {
						_bounds: dataset.bounds,
						src_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
					},
					#DOWNSAMPLE_FIELD_TEMPLATE & {
						_bounds: dataset.bounds,
						src_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
						_use_perlin_field: true
					},
					if dataset.contiguous {
						#DOWNSAMPLE_FIELD_TEMPLATE & {
							_bounds: dataset.bounds,
							src_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
						}
					}
					if dataset.contiguous {
						#DOWNSAMPLE_FIELD_TEMPLATE & {
							_bounds: dataset.bounds,
							src_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
							_use_perlin_field: true
						}
					}
				]
			}
		]
	}
}

#EXTRACT_DISPLACEMENT_STAGE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
	worker_resources: {
		"memory":     "8Gi"
	}
	worker_replicas:      100
	batch_gap_sleep_sec:  0.1
	do_dryrun_estimation: true
	local_test:           false
	worker_cluster_project: "zetta-research"
	worker_cluster_region: "us-east1"
	worker_cluster_name: "zutils-x3"
	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for key, dataset in #DATASETS {
				"@type": "mazepa.concurrent_flow"
				stages: [
					#FIELD_DIFF_TEMPLATE & {
						_bounds: dataset.bounds,
						dst_resolution: dataset.resolution
						_layer_name: key,
						_z_offset: 1
					},
					#FIELD_DIFF_TEMPLATE & {
						_bounds: dataset.bounds,
						dst_resolution: [dataset.resolution[0] * 2, dataset.resolution[1] * 2, dataset.resolution[2]]
						_layer_name: key,
						_z_offset: 1
					},
					if dataset.contiguous {
						#FIELD_DIFF_TEMPLATE & {
							_bounds: dataset.bounds,
							dst_resolution: dataset.resolution
							_layer_name: key,
							_z_offset: 2
						}
					}
					if dataset.contiguous {
						#FIELD_DIFF_TEMPLATE & {
							_bounds: dataset.bounds,
							dst_resolution: [dataset.resolution[0] * 2, dataset.resolution[1] * 2, dataset.resolution[2]]
							_layer_name: key,
							_z_offset: 2
						}
					}
				]
			}
		]
	}
}


#ENCODE_IMAGE_STAGE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231118"
	worker_resources: {
		"nvidia.com/gpu":     "1"
	}
	worker_replicas:      300
	batch_gap_sleep_sec:  0.1
	do_dryrun_estimation: true
	local_test:           false
	worker_cluster_project: "zetta-research"
	worker_cluster_region: "us-east1"
	worker_cluster_name: "zutils-x3"
	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for key, dataset in #DATASETS {
				"@type": "mazepa.concurrent_flow"
				stages: [
					for i in list.Range(0, 2, 1) {
						"@type": "mazepa.concurrent_flow"
						stages: [
							#ENCODE_IMG_TEMPLATE & {
								_bounds: dataset.bounds,
								_high_resolution: dataset.resolution
								_layer_name: key,
								_z_offset: 1
								_model: #MODELS[i]
							},
							#ENCODE_IMG_TEMPLATE & {
								_bounds: dataset.bounds,
								_high_resolution: dataset.resolution
								_layer_name: key,
								_z_offset: 1
								_use_perlin_field: true
								_model: #MODELS[i]
							},
							if dataset.contiguous {
								#ENCODE_IMG_TEMPLATE & {
									_bounds: dataset.bounds,
									_high_resolution: dataset.resolution
									_layer_name: key,
									_z_offset: 2
									_model: #MODELS[i]
								}
							}
							if dataset.contiguous {
								#ENCODE_IMG_TEMPLATE & {
									_bounds: dataset.bounds,
									_high_resolution: dataset.resolution
									_layer_name: key,
									_z_offset: 2
									_use_perlin_field: true
									_model: #MODELS[i]
								},
							}
						]
					}
				]
			}
		]
	}
}


[
  #COMPUTE_FIELD_STAGE,
  #WARP_IMAGE_STAGE,
  #DOWNSAMPLE_FIELD_STAGE,
  #EXTRACT_DISPLACEMENT_STAGE,
  #ENCODE_IMAGE_STAGE,
]