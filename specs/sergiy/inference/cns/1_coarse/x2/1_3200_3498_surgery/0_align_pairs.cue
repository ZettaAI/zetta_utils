// RUN ACED BLOCK

// INPUTS
#IMG_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/raw_img"

#ENC_PATH:           "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings_masked"
#INITIAL_FIELDS_DIR: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_rigid/v1/fields"
#DEFECTS_PATH:       "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/defect_mask"
#R2E_FIELD_PATH:     "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#TMP_PATH:           "gs://tmp_2s/yo/"

// MODELS
#BASE_ENCODER_PATH: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"
#MISD_MODEL_PATH:   "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr5.0_lr0.00001_z1z2_400-500_2910-2920_more_aligned_unet5_32_finetune_2/last.ckpt.static-2.0.0+cu117-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "coarse_z3300_3500_x0"

#FOLDER:          "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_surgery/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/pairwise_fields"
#FIELDS_INV_PATH: "\(#FOLDER)/pairwise_fields_inv"

#IMGS_WARPED_PATH:   "\(#FOLDER)/imgs_warped"
#MISALIGNMENTS_PATH: "\(#FOLDER)/misalignments"
#TISSUE_MASK_PATH:   "\(#FOLDER)/tissue_mask"

#CF_INFO_CHUNK: [512, 512, 1]

#Z_START: 3299
#Z_END:   3302

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [int, int, int] | *[2048, 2048, #Z_END]
	resolution: [512, 512, 45]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [50000, 60000, #Z_START + 1]
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
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		shrink_processing_chunk: true
		expand_bbox_processing:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		shrink_processing_chunk: true
		expand_bbox_processing:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       300
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
	bbox:        #ROI_BOUNDS
	stages:      #STAGES
	src_offset?: _
	tgt_offset?: _
	offset_resolution: [4, 4, 45]
	src: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
	}
	src_field: {
		"@type": "build_cv_layer"
		path:    _
		data_resolution: [256, 256, 45]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
			encoding:     "zfpc"
		}
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
			encoding:     "zfpc"
		}
		on_info_exists: "overwrite"
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
	bbox:           _
	dst_resolution: _ | *#STAGES[len(#STAGES)-1].dst_resolution
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
	bbox:           _
	src: {
		"@type": "build_ts_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size:     #CF_INFO_CHUNK
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
	bbox:           _
	_z_offset:      _
	src: {
		"@type": "build_ts_layer"
		path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)"
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
	bbox:           _
	_z_offset:      _
	src: {
		"@type": "build_ts_layer"
		path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)_enc"
	}
	tgt: {
		"@type": "build_ts_layer"
		path:    #ENC_PATH
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
	_bbox:   #ROI_BOUNDS
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.seq_flow"
					stages: [
						#CF_FLOW_TMPL & {
							dst: path:       "\(#FIELDS_PATH)/\(z_offset)"
							src_field: path: "\(#INITIAL_FIELDS_DIR)/\(z_offset)"
							tmp_layer_dir: "\(#FIELDS_PATH)/\(z_offset)/tmp"
							tgt_offset: [0, 0, z_offset]
							bbox: _bbox
						},
						#INVERT_FLOW_TMPL & {
							src: path: "\(#FIELDS_PATH)/\(z_offset)"
							dst: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
							bbox: _bbox
						},
						#WARP_FLOW_TMPL & {
							bbox: _bbox
							op: mode:  "img"
							dst: path: "\(#IMGS_WARPED_PATH)/\(z_offset)"
							src: path: #IMG_PATH
							src: index_procs: [
								{
									"@type": "VolumetricIndexTranslator"
									offset: [0, 0, z_offset]
									resolution: [4, 4, 45]
								},
							]
							field: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
						},
						#ENCODE_FLOW_TMPL & {
							bbox:      _bbox
							_z_offset: z_offset
						},
						#MISD_FLOW_TMPL & {
							bbox:      _bbox
							_z_offset: z_offset
						},
					]
				},
			]
		},
	]
}

#CREATE_TISSUE_MASK: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #ROI_BOUNDS
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: (src != 0).byte()"
	}
	processing_chunk_sizes: [[8 * 1024, 1024 * 8, 1]]
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution

	src: {
		"@type": "build_ts_layer"
		path:    #ENC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #TISSUE_MASK_PATH
		info_reference_path: src.path
		info_field_overrides: {
			data_type: "uint8"
		}
		on_info_exists: "overwrite"
	}
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
	_bbox:   #ROI_BOUNDS
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode:  "img" // not thresholded due to subhcunkable bug
								src: path: "\(#MISALIGNMENTS_PATH)/\(z_offset)"
								// src: read_procs: [
								//  {"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 20},
								// ]
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode:  "mask"
								src: path: #TISSUE_MASK_PATH
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode:  "field"
								src: path: "\(#FIELDS_PATH)/\(z_offset)"
								dst_resolution: [res, res, 45]
							}
						},
					]
				},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode:  "field"
								src: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
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
	"@type":                "mazepa.execute_on_gcp_with_sqs"
	worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x186"
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      10
	do_dryrun_estimation: true
	local_test:           false

	target: {
		"@type": "mazepa.seq_flow"
		stages: [
			#JOINT_OFFSET_FLOW,
			#CREATE_TISSUE_MASK,
			#DOWNSAMPLE_FLOW,
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
