import "math"

import "list"

#TMP_PATH: "gs://tmp_2w/prepare_cns"

#INITIAL_FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#BASE_FOLDER:        "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1"

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 425]
	end_coord: [int, int, int] | *[32768, 36864, 435]
	// end_coord: [32768, 32768, 3001]
	resolution: [32, 32, 45]
}

#ENCODER_MODELS: [
	{
		path: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [1, 1, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M4_conv1_unet3_lr0.0001_equi0.5_post1.6_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [2, 2, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M5_conv2_unet2_lr0.0001_equi0.5_post1.4_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [4, 4, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/4_M3_M6_conv3_unet1_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [8, 8, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M3_M7_conv4_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/epoch=0-step=1584-backup.ckpt.model.spec.json"
		res_change_mult: [16, 16, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M3_M8_conv5_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M4_M9_conv5_lr0.00002_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/7_M5_M10_conv5_lr0.0001_equi0.5_post1.06_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/6_M6_M11_conv5_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/6_M7_M12_conv5_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
]

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	// end_coord: [32768, 32768, 7010]
	end_coord: [32768, 36864, 7010]
	resolution: [32, 32, 45]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////  COPY INITIAL ////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#COPY_INITIAL_FIELD_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[2 * 1024, 2 * 1024, 1]]
	processing_crop_pads: [[0, 0, 0]]
	expand_bbox_processing: true
	dst_resolution: [256, 256, 45]
	bbox: #ROI_BOUNDS
	src: {
		"@type": "build_cv_layer"
		path:    #INITIAL_FIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#BASE_FOLDER)/field"
		info_reference_path: #INITIAL_FIELD_PATH
		//on_info_exists:      "overwrite"
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////  WARP  //////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#WARP_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [512, 512, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: _
	bbox:           #ROI_BOUNDS
	src: {
		"@type":    "build_ts_layer"
		path:       _
		read_procs: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
		path:    "\(#BASE_FOLDER)/field"
		data_resolution: [256, 256, 45]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: _
		//on_info_exists:      "overwrite"
		write_procs: _ | *[]
	}
}

#WARP_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#WARP_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/resin/ResinNet20221115_29k"
			src: read_procs: [
				{"@type": "compare", "@mode": "partial", mode: ">=", value: 48},
			]
			dst: path:                "\(#BASE_FOLDER)/resin_mask"
			dst: info_reference_path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/resin_mask"
			dst: write_procs: [
				{"@type": "to_uint8", "@mode": "partial"},
			]
			dst_resolution: [256, 256, 45]
			op: mode: "mask"
		},
		#WARP_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/defects/DefectNet20221114_50k"
			src: read_procs: [
				{"@type": "compare", "@mode": "partial", mode: ">=", value: 48},
			]
			dst: path: "\(#BASE_FOLDER)/defect_mask"
			dst_resolution: [64, 64, 45]
			dst: info_reference_path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
			dst: write_procs: [
				{"@type": "to_uint8", "@mode": "partial"},
			]
			op: mode: "mask"
		},
		#WARP_TMPL & {
			src: path:                "gs://zetta_lee_fly_cns_001_alignment_temp/rigid"
			dst: path:                "\(#BASE_FOLDER)/raw_img"
			dst: info_reference_path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img"
			dst_resolution: [32, 32, 45]
			op: mode: "img"
		},

	]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// DOWNSAMPLE//////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#DOWNSAMPLE_FLOW_TMPL: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	dst_resolution: _
	op: {
		"@type":         "InterpolateOperation"
		mode:            _
		res_change_mult: _ | *[2, 2, 1]
	}
	bbox: #ROI_BOUNDS
	src: {
		"@type":    "build_ts_layer"
		path:       _
		read_procs: _ | *[]
	}
	dst: {
		"@type": "build_cv_layer"
		path:    src.path
	}
}

#DOWNSAMPLE_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		{
			"@type": "mazepa.seq_flow"
			stages: [
				#DOWNSAMPLE_FLOW_TMPL & {
					op: mode:  "mask"
					src: path: "\(#BASE_FOLDER)/defect_mask"
					op: res_change_mult: [0.5, 0.5, 1]
					dst_resolution: [32, 32, 45]
				},
				for res in [128, 256, 512, 1024, 2048, 4096] {
					#DOWNSAMPLE_FLOW_TMPL & {
						op: mode:  "mask"
						src: path: "\(#BASE_FOLDER)/defect_mask"
						src: read_procs: [
							{"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 20},
						]
						dst_resolution: [res, res, 45]
					}
				},
			]
		},
		{
			"@type": "mazepa.seq_flow"
			stages: [
				for res in [64, 128, 256, 512, 1024, 2048, 4096] {
					#DOWNSAMPLE_FLOW_TMPL & {
						op: mode:  "img"
						src: path: "\(#BASE_FOLDER)/raw_img"
						dst_resolution: [res, res, 45]
					}
				},
			]
		},
	]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// ENCODING //////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

#ENC_INFO_CHUNK_SIZE: [1024, 1024, 1] // Will automatically get truncated if dataset becomes too small
#ENCODE_TASK_SIZE: [2048, 2048, 1]

#ENC_SCALES: [
	for i in list.Range(0, 10, 1) {
		let ds_factor = [math.Pow(2, i), math.Pow(2, i), 1]
		let vx_res = [ for j in [0, 1, 2] {#DATASET_BOUNDS.resolution[j] * ds_factor[j]}]
		let ds_offset = [ for j in [0, 1, 2] {
			__div(#DATASET_BOUNDS.start_coord[j], ds_factor[j])
		}]
		let ds_size = [ for j in [0, 1, 2] {
			__div((#DATASET_BOUNDS.end_coord[j] - ds_offset[j]), ds_factor[j])
		}]

		chunk_sizes: [[ for j in [0, 1, 2] {list.Min([#ENC_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
		resolution:   vx_res
		encoding:     "raw"
		key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
		voxel_offset: ds_offset
		size:         ds_size
	},
]

#ENCODE_FLOW_TMPL: {
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

	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 1, 1024 * 1, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	dst_resolution: [int, int, int]
	bbox: #ROI_BOUNDS
	src: {
		"@type": "build_ts_layer"
		path:    "\(#BASE_FOLDER)/raw_img"
	}
	dst: {
		"@type":               "build_cv_layer"
		path:                  "\(#BASE_FOLDER)/encodings"
		info_field_overrides?: _
		on_info_exists:        "overwite"
	}
}

#ENCODE_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#ENCODE_FLOW_TMPL & {
			op: fn: {
				"@type":    "BaseEncoder"
				model_path: #ENCODER_MODELS[0].path
			}
			op: res_change_mult: #ENCODER_MODELS[0].res_change_mult
			dst_resolution: #DATASET_BOUNDS.resolution
			dst: info_field_overrides: {
				type:         "image"
				num_channels: 1
				data_type:    "int8"
				scales:       #ENC_SCALES
			}
		},
		for i in list.Range(1, 10, 1) {
			let res_mult = [math.Pow(2, i), math.Pow(2, i), 1]
			let dst_res = [ for j in [0, 1, 2] {#DATASET_BOUNDS.resolution[j] * res_mult[j]}]

			#ENCODE_FLOW_TMPL & {
				op: fn: {
					"@type":     "BaseCoarsener"
					model_path:  #ENCODER_MODELS[i].path
					tile_pad_in: #ENCODE_FLOW_TMPL.op.crop_pad[0] * #ENCODER_MODELS[i].res_change_mult[0]
					tile_size:   1024
					ds_factor:   #ENCODER_MODELS[i].res_change_mult[0]
				}
				op: res_change_mult: #ENCODER_MODELS[i].res_change_mult
				dst_resolution: dst_res
			}
		},
	]
}

#MASK_ENCODINGS_FLOW_TMPL: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1]]
	dst_resolution: _
	fn: {
		"@type": "apply_mask_fn"
		"@mode": "partial"
	}
	bbox: #ROI_BOUNDS
	src: {
		"@type": "build_ts_layer"
		path:    "\(#BASE_FOLDER)/encodings"
	}
	masks: [
		{
			"@type": "build_ts_layer"
			path:    "\(#BASE_FOLDER)/defect_mask"
		},
	]
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#BASE_FOLDER)/encodings_masked"
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

#MASK_ENCODINGS_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
			#MASK_ENCODINGS_FLOW_TMPL & {
				dst_resolution: [res, res, 45]
			}
		},

	]
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// EXECUTION PARAMS ///////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#EXECUTE_TMPL: {
	"@type":                "mazepa.execute_on_gcp_with_sqs"
	worker_image:           "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x139"
	worker_resources:       _
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"
	worker_replicas:        _
	batch_gap_sleep_sec:    1.0
	local_test:             false
	target:                 _
}
#EXECUTE_ON_CPU: #EXECUTE_TMPL & {
	worker_resources: {
		memory: "18560Mi"
	}
	worker_replicas: 100
}

#EXECUTE_ON_GPU: #EXECUTE_TMPL & {
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas: 300
}

#COPY_INITIAL_FIELD: #EXECUTE_ON_CPU & {
	target: #COPY_INITIAL_FIELD_FLOW
}

#WARP: #EXECUTE_ON_CPU & {
	target: #WARP_FLOW
}

#DOWNSAMPLE: #EXECUTE_ON_CPU & {
	target: #DOWNSAMPLE_FLOW
}

#ENCODE: #EXECUTE_ON_GPU & {
	target: #ENCODE_FLOW
}

#MASK_ENCODINGS: #EXECUTE_ON_GPU & {
	target: #MASK_ENCODINGS_FLOW
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////  ORCHESTRATE /////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
[
	//#COPY_INITIAL_FIELD,
	//#WARP,
	//#DOWNSAMPLE,
	#ENCODE,
	#MASK_ENCODINGS,
]
