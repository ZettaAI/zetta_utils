#IMG_PATH_BASE:     "gs://hive-tomography/pilot11-tiles/refactor-test-0"
#DST_PATH_BASE:     "gs://hive-tomography/pilot11-tiles/refactor-test-0-enc"
#INTERMEDIARY_PATH: "gs://tmp_2w/hive-tomography/pilot11-tiles/refactor-test-0"

#IMG_RES: [4, 4, 1]

#IMG_SIZE: [786432, 262144, 111]

#MODELS: #GENERAL_ENC_MODELS

#OFFSETS: ["0_0", "0_1", "1_0", "1_1"]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [2048, 2048, 0]
	end_coord: [24576, 16384, 2]
	resolution: [16, 16, 1]
}

//#XY_CLAHE_RES: [4, 8, 16, 32, 64, 128, 256, 512]

//#XY_ENC_RES: []
#XY_CLAHE_RES: [16]
#XY_ENC_RES: [32, 64, 128, 256, 512, 1024, 2048]

#PROCESS_CROP_PAD: [16, 16, 0] // 16 pix was okay for 1um model

#SRC_TMPL: {
	"@type": "build_cv_layer"
	path:    _
}

#GCP_FLOW: {
	"@type":                "mazepa.execute_on_gcp_with_sqs"
	worker_cluster_region:  "us-east1"
	worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-test-montaging-refactor-10"
	worker_cluster_project: "zetta-research"
	worker_cluster_name:    "zutils-x3"
	worker_resources: {
		memory: "18560Mi" // sized for n1-highmem-4
	}
	worker_cluster_project: "zetta-research"
	worker_cluster_name:    "zutils-x3"
	worker_replicas:        100
	local_test:             false
	debug:                  false

	num_procs: 2
	target:    _
	semaphores_spec: {
		"read":  2
		"cpu":   2
		"cuda":  1
		"write": 2
	}
	do_dryrun_estimation: false
}
#GCP_FLOW & {
	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for offset in #OFFSETS for xy in #XY_ENC_RES {
				let model = #MODELS["\(xy)"]
				let bbox_ = #BBOX
				#ENC_FLOW_TMPL & {
					bbox: bbox_
					op: fn: model_path:  model.path
					op: fn: ds_factor:   model.res_change_mult[0]
					op: fn: tile_pad_in: model.res_change_mult[0] * #PROCESS_CROP_PAD[0]
					op: fn: tile_size:   model.res_change_mult[0] * model.process_chunk_sizes[1][0]
					op: res_change_mult: model.res_change_mult
					op: crop_pad:        #PROCESS_CROP_PAD
					op_kwargs: src: path: "\(#IMG_PATH_BASE)/\(offset)"
					dst: path: "\(#DST_PATH_BASE)/\(offset)"
					processing_chunk_sizes: model.process_chunk_sizes
					processing_crop_pads: [[0, 0, 0], #PROCESS_CROP_PAD]
					dst_resolution: [xy, xy, #IMG_RES[2]]
				}
			},
			for offset in #OFFSETS for xy in #XY_CLAHE_RES {
				"@type": "build_subchunkable_apply_flow"
				level_intermediaries_dirs: [#INTERMEDIARY_PATH, "file://."]
				bbox: #BBOX
				fn: {
					"@type":    "lambda"
					lambda_str: "lambda src:src"
				}
				fn_semaphores: ["cpu"]
				op_kwargs: {
					src: {
						"@type": "build_cv_layer"
						path:    "\(#IMG_PATH_BASE)/\(offset)"
					}
				}
				dst: #DST_TMPL
				dst: path: "\(#DST_PATH_BASE)/\(offset)"
				processing_chunk_sizes: [[2048, 2048, 1], [1024, 1024, 1]]
				processing_crop_pads: [[0, 0, 0], [512, 512, 0]]
				dst_resolution: [xy, xy, #IMG_RES[2]]
			},
		]
	}

}

#ENC_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    _
	//level_intermediaries_dirs: [null, "file://."]
	level_intermediaries_dirs: [#INTERMEDIARY_PATH, "file://."]
	// skip_intermediaries: true
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":     "BaseCoarsener"
			model_path:  _
			ds_factor:   _
			tile_pad_in: _
			tile_size:   _
		}
		fn_semaphores: ["cuda"]
		res_change_mult: _
		crop_pad:        _ | *#PROCESS_CROP_PAD
	}
	op_kwargs: {
		src: #SRC_TMPL
	}
	dst_resolution:         _
	processing_chunk_sizes: _
	processing_crop_pads:   _
	// expand_bbox_resolution: true
	dst: #DST_TMPL
}

#GENERAL_ENC_MODELS: {
	"32": {
		path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [1, 1, 1] //
		process_chunk_sizes: [[2048, 2048, 1], [1024, 1024, 1]]
	}
	"64": {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [2, 2, 1] //
		process_chunk_sizes: [[1024, 1024, 1], [512, 512, 1]]
	}
	"128": {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [4, 4, 1] //
		process_chunk_sizes: [[1024, 1024, 1], [512, 512, 1]]
	}
	"256": {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [8, 8, 1] //
		process_chunk_sizes: [[512, 512, 1], [256, 256, 1]]
	}
	"512": {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [16, 16, 1] // 2
		process_chunk_sizes: [[512, 512, 1], [128, 128, 1]]
	}
	"1024": {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M3_M8_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		process_chunk_sizes: [[256, 256, 1], [64, 64, 1]]
	}
	"2048": {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.2_M4_M9_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		process_chunk_sizes: [[128, 128, 1], [32, 32, 1]]
	}
}

#DST_TMPL: {
	"@type": "build_cv_layer"
	path:    _
	info_add_scales_ref: {
		resolution: #IMG_RES
		size:       #IMG_SIZE
		chunk_sizes: [[512, 512, 1]]
		encoding: "raw"
		voxel_offset: [0, 0, -100]
	}
	info_add_scales: [
		for xy in #XY_ENC_RES {
			[xy, xy, #IMG_RES[2]]
		},
		for xy in #XY_CLAHE_RES {
			[xy, xy, #IMG_RES[2]]
		},
	]
	info_add_scales_mode: "merge"
	info_field_overrides: {
		type:         "image"
		num_channels: 1
		data_type:    "int8"
		type:         "image"
		voxel_offset: [0, 0, -100]
	}
	on_info_exists: "overwrite"
	write_procs: [{"@type": "apply_clahe"}]
}
