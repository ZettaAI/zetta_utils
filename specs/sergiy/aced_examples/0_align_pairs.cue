// R 2UN ACED BLOCK
// INPUTS
#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced"

#IMG_PATH:     "\(#BASE_FOLDER)/coarse_x1/raw_img"
#DEFECTS_PATH: "\(#BASE_FOLDER)/coarse_x1/defect_mask"

//#ENC_PATH:      "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"
#ENC_PATH: "\(#BASE_FOLDER)/coarse_x1/encodings_masked"
#TMP_PATH: "gs://tmp_2w/temporary_layers"

// MODELS
#BASE_ENCODER_PATH: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"
#MISD_MODEL_PATH:   "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr5.0_lr0.00001_z1z2_400-500_2910-2920_more_aligned_unet5_32_finetune_2/last.ckpt.static-2.0.0+cu117-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "try_x0"

#FOLDER:          "\(#BASE_FOLDER)/med_x1/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

#TISSUE_MASK_PATH: "\(#BASE_FOLDER)/tissue_mask"

//#MATCH_OFFSETS_BASE: "\(#FOLDER)/match_offsets_z\(#Z_START)_\(#Z_END)"
#MATCH_OFFSET_BASE: "\(#FOLDER)/match_offsets_v0_z"

//#BASE_INFO_CHUNK: [128, 128, 1]
#BASE_INFO_CHUNK: [512, 512, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [2048, 2048, 100]
	resolution: [512, 512, 45]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [50000, 60000, 1000]
	scale_bar_nm: 30000
	layers: [
		["precoarse_img", "image", "precomputed://\(#IMG_PATH)"],
		["-1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/-1"],
	]
}

#STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [2048, 2048, 45]
		fn: {
			sm:       100
			num_iter: 1000
			lr:       0.015
		}
		shrink_processing_chunk: true
		expand_bbox_processing:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       100
			num_iter: 1000
			lr:       0.015
		}
		shrink_processing_chunk: true
		expand_bbox_processing:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       50
			num_iter: 700
			lr:       0.015
		}
		shrink_processing_chunk: true
		expand_bbox_processing:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 45]

		fn: {
			sm:       50
			num_iter: 700
			lr:       0.03
		}
		shrink_processing_chunk: true
		expand_bbox_processing:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [128, 128, 45]

		fn: {
			sm:       10
			num_iter: 500
			lr:       0.05
		}

	},
	#STAGE_TMPL & {
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       10
			num_iter: 1000
			lr:       0.015
		}

	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       10
			num_iter: 700
			lr:       0.015
		}

	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 45]

		fn: {
			sm:       10
			num_iter: 700
			lr:       0.03
		}

	},
	#STAGE_TMPL & {
		dst_resolution: [128, 128, 45]

		fn: {
			sm:       10
			num_iter: 500
			lr:       0.05
		}

	},
	#STAGE_TMPL & {
		dst_resolution: [64, 64, 45]

		fn: {
			sm:       10
			num_iter: 300
			lr:       0.1
		}

	},

	#STAGE_TMPL & {
		dst_resolution: [32, 32, 45]

		fn: {
			sm:       10
			num_iter: 200
			lr:       0.1
		}

	},
]

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _

	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	expand_bbox_processing:             bool | *true
	shrink_processing_chunk: bool | *false

	fn: {
		"@type":  "align_with_online_finetuner"
		"@mode":  "partial"
		sm:       _
		num_iter: _
		lr?:      _
	}
}

#CF_FLOW_TMPL: {
	"@type":     "build_compute_field_multistage_flow"
	bbox:        #BBOX
	stages:      #STAGES
	src_offset?: _
	tgt_offset?: _
	offset_resolution: [4, 4, 45]
	src: {
		"@type": "build_ts_layer"
		path:    #ENC_PATH
	}
	tgt: {
		"@type": "build_ts_layer"
		path:    #ENC_PATH
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _
		//info_reference_path: #IMG_PATH
		//info_field_overrides: {
		//num_channels: 2
		//data_type:    "float32"
		//}
		//info_chunk_size: #BASE_INFO_CHUNK
		//on_info_exists:  "overwrite"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type": "build_cv_layer"
		"@mode": "partial"
		//info_reference_path: dst.path
		//info_reference_path: #IMG_PATH
		//info_field_overrides: {
		//  num_channels: 2
		//  data_type:    "float32"
		// }
		//info_chunk_size: #BASE_INFO_CHUNK
		//on_info_exists:  "overwrite"
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	//chunk_size: [512, 512, 1]
	bbox:           #BBOX
	dst_resolution: _ | *#STAGES[len(#STAGES)-1].dst_resolution
	op_kwargs: {
		src: {
			"@type":      "build_ts_layer"
			path:         _
			read_procs?:  _
			index_procs?: _ | *[]
		}
		field: {
			"@type":            "build_ts_layer"
			path:               _
			data_resolution:    _ | *null
			interpolation_mode: "field"
		}

	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		write_procs?:        _
		index_procs?:        _ | *[]
	}
}

#INVERT_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	op_kwargs: {
		src: {
			"@type": "build_ts_layer"
			path:    _
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: op_kwargs.src.path
		info_chunk_size:     #BASE_INFO_CHUNK
		on_info_exists:      "overwrite"
	}
}

#ENCODE_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "BaseEncoder"
		model_path: #BASE_ENCODER_PATH
	}
	expand_bbox_processing: true

	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 1, 1024 * 1, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	_z_offset:      _
	op_kwargs: {
		src: {
			"@type": "build_ts_layer"
			path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)"
		}

	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#IMGS_WARPED_PATH)/\(_z_offset)_enc"
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		info_field_overrides: {
			data_type: "int8"
		}
	}
}

#MISD_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "MisalignmentDetector"
		model_path: #MISD_MODEL_PATH
	}
	expand_bbox_processing: true

	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	_z_offset:      _
	op_kwargs: {
		src: {
			"@type": "build_ts_layer"
			path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)_enc"
		}
		tgt: {
			"@type": "build_ts_layer"
			path:    #ENC_PATH
		}

	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#MISALIGNMENTS_PATH)/\(_z_offset)"
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		write_procs: [

		]
	}
}

#Z_OFFSETS: [-1, -2]
#JOINT_OFFSET_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.sequential_flow"
					stages: [
						#CF_FLOW_TMPL & {
							dst: path: "\(#FIELDS_PATH)/\(z_offset)"
							tmp_layer_dir: "\(#FIELDS_PATH)/\(z_offset)/tmp"
							tgt_offset: [0, 0, z_offset]
						},
						#INVERT_FLOW_TMPL & {
							op_kwargs: src: path: "\(#FIELDS_PATH)/\(z_offset)"
							dst: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
						},
						#WARP_FLOW_TMPL & {
							op: mode:  "img"
							dst: path: "\(#IMGS_WARPED_PATH)/\(z_offset)"
							op_kwargs: {
								src: path: #IMG_PATH
								src: index_procs: [
									{
										"@type": "VolumetricIndexTranslator"
										offset: [0, 0, z_offset]
										resolution: [4, 4, 45]
									},
								]
								field: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
							}
						},
						#ENCODE_FLOW_TMPL & {
							_z_offset: z_offset
						},
						#MISD_FLOW_TMPL & {
							_z_offset: z_offset
						},
					]
				},
			]
		},
	]
}

#DOWNSAMPLE_FLOW_TMPL: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	dst_resolution: _
	op: {
		"@type": "InterpolateOperation"
		mode:    _
		res_change_mult: [2, 2, 1]
	}
	bbox: #BBOX
	op_kwargs: {
		src: {
			"@type":    "build_ts_layer"
			path:       _
			read_procs: _ | *[]
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    op_kwargs.src.path
	}
}

#DOWNSAMPLE_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.sequential_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								op: mode: "img" // not thresholded due to subhcunkable bug
								op_kwargs: src: path: "\(#MISALIGNMENTS_PATH)/\(z_offset)"
								// src: read_procs: [
								//  {"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 20},
								// ]
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
				{
					"@type": "mazepa.sequential_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								op: mode: "mask"
								op_kwargs: src: path: #TISSUE_MASK_PATH
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
				{
					"@type": "mazepa.sequential_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								op: mode: "field"
								op_kwargs: src: path: "\(#FIELDS_PATH)/\(z_offset)"
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
				{
					"@type": "mazepa.sequential_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								op: mode: "field"
								op_kwargs: src: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
			]
		},
	]
}

#RUN_INFERENCE: {
	"@type": "mazepa.execute_on_gcp_with_sqs"
	//worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x140"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x163"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	checkpoint:             "gs://zetta_utils_runs/sergiy/exec-smart-crouching-turtle-of-chaos/2023-04-22_172435_972.zstd"
	worker_replicas:        200
	batch_gap_sleep_sec:    1
	do_dryrun_estimation:   true
	local_test:             false
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"

	target: {
		"@type": "mazepa.sequential_flow"
		stages: [
			#JOINT_OFFSET_FLOW,
			#DOWNSAMPLE_FLOW,
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
