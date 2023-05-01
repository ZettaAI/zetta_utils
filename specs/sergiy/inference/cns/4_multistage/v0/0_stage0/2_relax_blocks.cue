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

#AFIELD_PATH:      "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH: "\(#FOLDER)/img_aligned\(#RELAXATION_SUFFIX)"
#IMG_MASK_PATH:    "\(#FOLDER)/img_mask\(#RELAXATION_SUFFIX)"
#AFF_MASK_PATH:    "\(#FOLDER)/aff_mask\(#RELAXATION_SUFFIX)"

#IMG_ALIGNED_MASKED_PATH: "\(#FOLDER)/img_aligned_masked\(#RELAXATION_SUFFIX)"

//#MATCH_OFFSETS_BASE: "\(#FOLDER)/match_offsets_z\(#Z_START)_\(#Z_END)"
#MATCH_OFFSET_BASE: "\(#FOLDER)/match_offsets_v1_z"

//#BASE_INFO_CHUNK: [128, 128, 1]
#BASE_INFO_CHUNK: [512, 512, 1]
#RELAX_OUTCOME_CHUNK: [32, 32, 1]
#RELAXATION_FIX:       "both"
#RELAXATION_ITER:      8000
#RELAXATION_LR:        1e-3
#RELAXATION_GRAD_CLIP: 0.01

#RELAXATION_RIG: 0.5

//#Z_END:   746
#DEBUG_SUFFIX: ""

//#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"
#RELAXATION_SUFFIX: "_1024nm_try_x14_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_lr\(#RELAXATION_LR)_clip\(#RELAXATION_GRAD_CLIP)\(#DEBUG_SUFFIX)"
#RELAXATION_RESOLUTION: [1024, 1024, 45]
#BLOCKS: [
	{_z_start: 0, _z_end:    452},
	{_z_start: 451, _z_end:  1350},
	{_z_start: 1349, _z_end: 2251},
	//{_z_start: 2250, _z_end: 3155},
]

#BBOX_TMPL: {
	"@type":  "BBox3D.from_coords"
	_z_start: int
	_z_end:   int
	start_coord: [0, 0, _z_start]
	end_coord: [2048, 2048, _z_end]
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
		//["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX) -2", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
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
		shrink_processing_chunk: true
		expand_bbox:             false
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 45]

		fn: {
			sm:       50
			num_iter: 700
			lr:       0.03
		}
		shrink_processing_chunk: true
		expand_bbox:             false
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
	bbox:        _
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

#NAIVE_MISD_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "naive_misd"
		"@mode": "partial"
	}
	processing_chunk_sizes: [[2048, 2048, 1]]
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           _
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
	expand_bbox: true
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
	expand_bbox: true

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
	_bbox:   _
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.seq_flow"
					stages: [
						#CF_FLOW_TMPL & {
							dst: path: "\(#FIELDS_PATH)/\(z_offset)"
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
						}
						// #NAIVE_MISD_FLOW & {
						//  _z_offset: z_offset
						// },,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
					]
				},
			]
		},
	]
}

#CREATE_TISSUE_MASK: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    _
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

#MATCH_OFFSETS_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "AcedMatchOffsetOp"
	}
	bbox: _

	processing_chunk_sizes: [[64, 64, bbox._z_end - bbox._z_start]]
	processing_crop_pads: [[32, 32, 0]]
	dst_resolution: #RELAXATION_RESOLUTION
	max_dist:       2

	tissue_mask: {
		"@type": "build_ts_layer"
		path:    #TISSUE_MASK_PATH
	}
	misalignment_masks: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#MISALIGNMENTS_PATH)/\(offset)"
				read_procs: [
					{
						"@type": "compare"
						"@mode": "partial"
						mode:    ">="
						value:   80
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
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
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)\(bbox._z_start)_\(bbox._z_end)"
	dst: {
		"@type": "build_volumetric_layer_set"
		layers: {
			match_offsets: {
				"@type":             "build_cv_layer"
				path:                match_offsets_path
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			img_mask: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/img_mask"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			aff_mask: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/aff_mask"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			sector_length_before: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/sl_before"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			sector_length_after: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/sl_after"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
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
	dst_resolution: #RELAXATION_RESOLUTION
	bbox:           _

	processing_chunk_sizes: [[32, 32, bbox._z_end - bbox._z_start]]
	max_reduction_chunk_sizes: [32, 32, bbox._z_end - bbox._z_start]
	processing_crop_pads: [[32, 32, 0]]
	processing_blend_pads: [[8, 8, 0]]
	level_intermediaries_dirs: [#TMP_PATH]
	//              processing_chunk_sizes: [[32, 32, #Z_END - #Z_START], [28, 28, #Z_END - #Z_START]]
	//              max_reduction_chunk_sizes: [128, 128, #Z_END - #Z_START]
	//              processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	//              processing_blend_pads: [[12, 12, 0], [12, 12, 0]]
	//level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	fix:                     #RELAXATION_FIX
	num_iter:                #RELAXATION_ITER
	lr:                      #RELAXATION_LR
	grad_clip:               #RELAXATION_GRAD_CLIP
	rigidity_weight:         #RELAXATION_RIG
	min_rigidity_multiplier: 0.001

	rigidity_masks: {
		"@type": "build_ts_layer"
		path:    #DEFECTS_PATH
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "!="
				value:   0
			},
			{
				"@type": "filter_cc"
				"@mode": "partial"
				mode:    "keep_large"
				thr:     200
			},
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "=="
				value:   0
			},
		]
	}
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)\(bbox._z_start)_\(bbox._z_end)"
	match_offsets: {
		"@type": "build_ts_layer"
		path:    match_offsets_path

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
		info_chunk_size: #RELAX_OUTCOME_CHUNK
		on_info_exists:  "overwrite"
	}
}

#DOWNSAMPLE_FLOW_TMPL: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	dst_resolution: _
	op: {
		"@type": "InterpolateOperation"
		mode:    _
		res_change_mult: [2, 2, 1]
	}
	bbox: _
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
	_bbox:   _
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

#POST_ALIGN_FLOW: {
	_bbox:   _
	"@type": "mazepa.concurrent_flow"
	stages: [
		#WARP_FLOW_TMPL & {
			bbox: _bbox
			op: mode:    "img"
			src: path:   #IMG_PATH
			field: path: #AFIELD_PATH
			dst: path:   #IMG_ALIGNED_PATH
			dst_resolution: [32, 32, 45]
			field: data_resolution: #RELAXATION_RESOLUTION
		}
		// #WARP_FLOW_TMPL & {
		//  op: mode:    "mask"
		//  src: path:   "\(#MATCH_OFFSETS_PATH)/img_mask"
		//  field: path: #AFIELD_PATH
		//  dst: path:   #IMG_MASK_PATH
		//  dst_resolution: [32, 32, 45]
		//  field: data_resolution: #RELAXATION_RESOLUTION
		// },
		// #WARP_FLOW_TMPL & {
		//  op: mode:    "mask"
		//  src: path:   "\(#MATCH_OFFSETS_PATH)/aff_mask"
		//  field: path: #AFIELD_PATH
		//  dst: path:   #AFF_MASK_PATH
		//  dst_resolution: [32, 32, 45]
		//  field: data_resolution: #RELAXATION_RESOLUTION
		// },,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

	]
}

#RUN_INFERENCE: {
	"@type":                "mazepa.execute_on_gcp_with_sqs"
	worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x184"
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"
	//checkpoint:   "gs://zetta_utils_runs/sergiy/exec-macho-giraffe-of-ideal-domination/2023-04-24_114132_1249.zstd"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:        80
	batch_gap_sleep_sec:    1
	do_dryrun_estimation:   true
	local_test:             false
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"

	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for block in #BLOCKS {
				let bbox = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
				"@type": "mazepa.seq_flow"
				stages: [
					// #JOINT_OFFSET_FLOW & {
					//  _bbox: bbox
					// },
					// #CREATE_TISSUE_MASK & {
					//  'bbox': bbox
					// },
					// #DOWNSAMPLE_FLOW & {
					//  _bbox: bbox
					// }
					#MATCH_OFFSETS_FLOW & {'bbox': bbox},
					#RELAX_FLOW & {'bbox':         bbox},
					#POST_ALIGN_FLOW & {_bbox:     bbox},
				]
			},
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
