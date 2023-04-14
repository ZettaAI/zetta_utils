// R 2UN ACED BLOCK
// INPUTS
#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced"

#IMG_PATH:     "\(#BASE_FOLDER)/coarse_x0/raw_img"
#DEFECTS_PATH: "\(#BASE_FOLDER)/coarse_x0/defect_mask"

//#ENC_PATH:      "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"
#ENC_PATH: "\(#BASE_FOLDER)/coarse_x0/encodings_masked"
#TMP_PATH: "gs://tmp_2w/temporary_layers"

// MODELS
#BASE_ENCODER_PATH: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"
#MISD_MODEL_PATH:   "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr5.0_lr0.00001_z1z2_400-500_2910-2920_more_aligned_unet5_32_finetune_2/last.ckpt.static-2.0.0+cu117-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "try_x3"

#FOLDER:          "\(#BASE_FOLDER)/med_x0/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

#AFIELD_PATH:             "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH:        "\(#FOLDER)/img_aligned\(#RELAXATION_SUFFIX)"
#IMG_MASK_PATH:           "\(#FOLDER)/img_mask\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_MASKED_PATH: "\(#FOLDER)/img_aligned_masked\(#RELAXATION_SUFFIX)"

#MATCH_OFFSETS_PATH: "\(#FOLDER)/match_offsets_z\(#Z_START)_\(#Z_END)"
//#MATCH_OFFSETS_PATH: "\(#FOLDER)/match_offsets_x0"

#BASE_INFO_CHUNK: [128, 128, 1]
#RELAXATION_FIX:  "both"
#RELAXATION_ITER: 500
#RELAXATION_LR:   0.3

#RELAXATION_RIG: 20

//BLOCK 0
#Z_START: 429

#Z_END: 529

//#Z_END: 810

//#Z_START: 599
//#Z_END:   746

//#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"
#RELAXATION_SUFFIX: "_try_x4_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_END]
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
		//["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX) -2", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
	]
}

#NOT_FIRST_SECTION_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START + 1]
	end_coord: [ 2048, 2048, #Z_END]
	resolution: [512, 512, 45]
}

#FIRST_SECTION_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_START + 1]
	resolution: [512, 512, 45]
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
		expand_bbox:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       100
			num_iter: 1000
			lr:       0.015
		}
		shrink_processing_chunk: true
		expand_bbox:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       50
			num_iter: 700
			lr:       0.015
		}
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

	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 4, 1024 * 1, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	expand_bbox:             bool | *true
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
	bbox:        #NOT_FIRST_SECTION_BBOX
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
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
		}
		info_chunk_size: #BASE_INFO_CHUNK
		on_info_exists:  "overwrite"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
		}
		info_chunk_size: #BASE_INFO_CHUNK
		on_info_exists:  "overwrite"
	}
}

#NAIVE_MISD_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "naive_misd"
		"@mode": "partial"
	}
	processing_chunk_sizes: [[2048, 2048, 1]]
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	_z_offset:      _
	src: {
		"@type": "build_ts_layer"
		path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)"
	}
	tgt: {
		"@type": "build_ts_layer"
		path:    #IMG_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#MISALIGNMENTS_PATH)/\(_z_offset)"
		info_reference_path: #IMG_PATH
		info_chunk_size:     #BASE_INFO_CHUNK
		on_info_exists:      "overwrite"
		write_procs: [
		]
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	expand_bbox: true
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 4, 1024 * 1, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	//chunk_size: [512, 512, 1]
	bbox:           #BBOX
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	src: {
		"@type":      "build_ts_layer"
		path:         _
		read_procs?:  _
		index_procs?: _ | *[]
	}
	field: {
		"@type": "build_ts_layer"
		path:    _
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

	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 4, 1024 * 1, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	src: {
		"@type": "build_ts_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
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
	expand_bbox: true

	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 4, 1024 * 1, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	_z_offset:      _
	src: {
		"@type": "build_ts_layer"
		path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)"
	}
	dst: {
		"@type": "build_cv_layer"
		path:    "\(#IMGS_WARPED_PATH)/\(_z_offset)_enc"
		//path:                #MATCH_OFFSETS_PATH
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
	expand_bbox: true

	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 4, 1024 * 1, 1]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
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
		"@type": "build_cv_layer"
		path:    "\(#MISALIGNMENTS_PATH)/\(_z_offset)"
		//path:                #MATCH_OFFSETS_PATH
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		write_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    ">="
				value:   127
			},
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
}

#Z_OFFSETS: [-1]
#JOINT_OFFSET_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.seq_flow"
					stages: [
						#INVERT_FLOW_TMPL & {
							src: path: "\(#FIELDS_PATH)/\(z_offset)"
							dst: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
						},
					]
				},
			]
		},
	]
}

#MATCH_OFFSETS_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX
	op: {
		"@type": "AcedMatchOffsetOp"
	}
	processing_chunk_sizes: [[256, 256, #Z_END - #Z_START]]
	processing_crop_pads: [[32, 32, 0]]
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	max_dist:       2

	tissue_mask: {
		"@type": "build_ts_layer"
		path:    #ENC_PATH
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "!="
				value:   0
			},

		]
	}
	misalignment_masks: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#MISALIGNMENTS_PATH)/\(offset)"
			}
		}
	}
	pairwise_fields: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#FIELDS_PATH)/\(offset)"
			}
		}
	}
	pairwise_fields_inv: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#FIELDS_INV_PATH)/\(offset)"
			}
		}
	}
	dst: {
		"@type": "build_volumetric_layer_set"
		layers: {
			match_offsets: {
				"@type":             "build_cv_layer"
				path:                #MATCH_OFFSETS_PATH
				info_reference_path: #IMG_PATH
				info_chunk_size:     #BASE_INFO_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			img_mask: {
				"@type":             "build_cv_layer"
				path:                "\(#MATCH_OFFSETS_PATH)/img_mask"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #BASE_INFO_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			aff_mask: {
				"@type":             "build_cv_layer"
				path:                "\(#MATCH_OFFSETS_PATH)/aff_mask"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #BASE_INFO_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
		}
	}
}

#RELAX_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "AcedRelaxationOp"
	}
	expand_bbox:    true
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	processing_chunk_sizes: [[128, 128, #Z_END - #Z_START], [72, 72, #Z_END - #Z_START]]
	max_reduction_chunk_sizes: [128, 128, #Z_END - #Z_START]
	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	processing_blend_pads: [[8, 8, 0], [8, 8, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	fix:             #RELAXATION_FIX
	num_iter:        #RELAXATION_ITER
	lr:              #RELAXATION_LR
	rigidity_weight: #RELAXATION_RIG

	rigidity_masks: {
		"@type": "build_ts_layer"
		path:    #DEFECTS_PATH
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "=="
				value:   0
			},

		]
	}
	match_offsets: {
		"@type": "build_ts_layer"
		path:    #MATCH_OFFSETS_PATH
		//info_reference_path: #IMG_PATH
		on_info_exists: "overwrite"
	}
	pfields: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#FIELDS_PATH)/\(offset)"
			}
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFIELD_PATH
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
		}
		info_chunk_size: #BASE_INFO_CHUNK
		on_info_exists:  "overwrite"
	}
}

#POST_ALIGN_FLOW: {
	"@type": "mazepa.seq_flow"
	stages: [
		#WARP_FLOW_TMPL & {
			op: mode:    "img"
			src: path:   #IMG_PATH
			field: path: #AFIELD_PATH
			dst: path:   #IMG_ALIGNED_PATH
		}
		// #WARP_FLOW_TMPL & {
		//  op: mode:    "mask"
		//  src: path:   "\(#MATCH_OFFSETS_PATH)/img_mask"
		//  field: path: #AFIELD_PATH
		//  dst: path:   #IMG_MASK_PATH
		// },,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
	]
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x136"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      100
	batch_gap_sleep_sec:  1
	do_dryrun_estimation: true
	local_test:           true

	target: {
		"@type": "mazepa.seq_flow"
		stages: [
			#JOINT_OFFSET_FLOW,
			//#MATCH_OFFSETS_FLOW,
			//#RELAX_FLOW,
			//#POST_ALIGN_FLOW,
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
