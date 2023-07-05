// RUN ACED BLOCK

// INPUTS
#IMG_PATH:           "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/raw_img"
#BASE_ENC_PATH:      "TODO"
#ENC_PATH:           "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/encodings_masked"
#INITIAL_FIELDS_DIR: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_rigid/v1/fields"
#DEFECTS_PATH:       "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/defect_mask"
#R2E_FIELD_PATH:     "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"

// MODELS
#BASE_ENC_MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"
#MISD_MODEL_PATH:     "gs://sergiy_exp/training_artifacts/aced_misd/zm1_zm2_thr1.0_scratch_large_custom_dset_x2/checkpoints/epoch=2-step=1524.ckpt.static-1.12.1+cu102-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "coarse_z3300_3500_x0"

#FOLDER:          "gs://sergiy_exp/aced/demo_x0/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/pairwise_fields"
#FIELDS_INV_PATH: "\(#FOLDER)/pairwise_fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

#MATCH_OFFSETS_PATH: "\(#FOLDER)/match_offsets"

#AFIELD_PATH:             "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH:        "\(#FOLDER)/img_aligned\(#RELAXATION_SUFFIX)"
#IMG_MASK_PATH:           "\(#FOLDER)/img_mask\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_MASKED_PATH: "\(#FOLDER)/img_aligned_masked\(#RELAXATION_SUFFIX)"

#CF_INFO_CHUNK: [512, 512, 1]
#AFIELD_INFO_CHUNK: [512, 512, 1]
#RELAXATION_FIX:  "both"
#RELAXATION_ITER: 500
#RELAXATION_LR:   0.3
#RELAXATION_RIG:  20

#Z_START:           3300
#Z_END:             3500
#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)_r2e_surgery_x0"

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
		["+1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/+1"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
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
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		chunk_size: [512, 512, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		chunk_size: [512, 512, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		chunk_size: [1024, 1024, 1]
	},
]

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	chunk_size:     _
	fn: {
		"@type":  "align_with_online_finetuner"
		"@mode":  "partial"
		sm:       _
		num_iter: _
		lr?:      _
	}
	crop_pad: [64, 64, 0]
}

#CF_FLOW_TMPL: {
	"@type":   "build_compute_field_multistage_flow"
	bbox:      #NOT_FIRST_SECTION_BBOX
	stages:    #STAGES
	_z_offset: _
	tgt_offset: [0, 0, _z_offset]
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
		path:    "\(#INITIAL_FIELDS_DIR)/\(_z_offset)"
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

#INVERT_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[64, 64, 0]]
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		on_info_exists:      "overwrite"
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	processing_crop_pads: [[256, 256, 0]]
	processing_chunk_sizes: [[2048, 2048, 1]]
	//chunk_size: [512, 512, 1]
	bbox:           #BBOX
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	src: {
		"@type":      "build_cv_layer"
		path:         _
		read_procs?:  _
		index_procs?: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
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

#MISD_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "MisalignmentDetector"
		model_path: #MISD_MODEL_PATH
	}
	processing_crop_pads: [[32, 32, 0]]
	processing_chunk_sizes: [[2048, 2048, 1]]
	bbox: #BBOX
	dst_resolution: [32, 32, 45]

	src: {
		"@type": "build_cv_layer"
		path:    #BASE_ENC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
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
					"@type": "mazepa.seq_flow"
					stages: [
						#CF_FLOW_TMPL & {
							dst: path: "\(#FIELDS_PATH)/\(z_offset)"
							tmp_layer_dir: "\(#FIELDS_PATH)/\(z_offset)/tmp"
							_z_offset:     z_offset
						},
						#INVERT_FLOW_TMPL & {
							src: path: "\(#FIELDS_PATH)/\(z_offset)"
							dst: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
						},
						#WARP_FLOW_TMPL & {
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
					]
				},

				//{
				// "@type": "mazepa.seq_flow"
				// stages: [
				//  #CF_FLOW_TMPL & {
				//   dst: path: "\(#FIELDS_BWD_PATH)/\(z_offset)"
				//   tmp_layer_dir: "\(#FIELDS_BWD_PATH)/\(z_offset)/tmp"
				//   src_offset: [0, 0, z_offset]
				//  },
				//  #WARP_FLOW_TMPL & {
				//   mode: "img"
				//   src: path: #IMG_PATH
				//   src: index_procs: [
				//    {
				//     "@type": "VolumetricIndexTranslator"
				//     offset: [0, 0, z_offset]
				//     resolution: [4, 4, 45]
				//    },
				//   ]
				//   field: path: "\(#FIELDS_BWD_PATH)/\(z_offset)"
				//   dst: path:   "\(#IMGS_WARPED_PATH)/\(z_offset)"
				//  },
				// ]
				//},,
			]
		},
	]
}

#RELAX_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "AcedRelaxationOp"
	}
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	expand_bbox:    true
	processing_chunk_sizes: [[512, 512, #Z_END - #Z_START]]
	processing_crop_pads: [[32, 32, 0]]
	level_intermediaries_dirs: ["file://~/.zutil/tmp"]
	//processing_chunk_sizes: [[512, 512, #Z_END - #Z_START], [256, 256, #Z_END - #Z_START]]
	//processing_crop_pads: [[0, 0, 0], [32, 32, 0]]
	//level_intermediaries_dirs: ["file://~/.zutils/tmp", "file://~/.zutils/tmp"]

	fix:             #RELAXATION_FIX
	num_iter:        #RELAXATION_ITER
	lr:              #RELAXATION_LR
	rigidity_weight: #RELAXATION_RIG

	first_section_fix_field: {
		"@type": "build_cv_layer"
		path:    #R2E_FIELD_PATH
		data_resolution: [256, 256, 45]
		interpolation_mode: "field"
	}
	last_section_fix_field: first_section_fix_field
	rigidity_masks: {
		"@type": "build_cv_layer"
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
		"@type": "build_cv_layer"
		path:    #MATCH_OFFSETS_PATH
		//info_reference_path: #IMG_PATH
		//on_info_exists: "expect_same"
	}
	pfields: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_cv_layer"
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
			encoding:     "zfpc"
		}
		info_chunk_size: #AFIELD_INFO_CHUNK
		on_info_exists:  "overwrite"
	}
}

#MISD_TMP_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "naive_misd"
		"@mode": "partial"
	}
	processing_chunk_sizes: [[2048, 2048, 1]]
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	bbox:           #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    "\(#IMGS_WARPED_PATH)/-1"
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #IMG_PATH
	}
	dst: {
		"@type": "build_cv_layer"
		//path:                "\(#MISALIGNMENTS_PATH)/-1"
		path:                #MATCH_OFFSETS_PATH
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		write_procs: [
			{
				"@type": "coarsen_mask"
				"@mode": "partial"
				width:   1
			},
			{
				"@type": "to_float32"
				"@mode": "partial"
			},
			{
				"@type": "add"
				"@mode": "partial"
				value:   -1
			},
			{
				"@type": "multiply"
				"@mode": "partial"
				value:   -1
			},
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
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
		// },,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
	]
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x114"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      8
	batch_gap_sleep_sec:  1
	do_dryrun_estimation: true
	local_test:           false

	target: {
		"@type": "mazepa.seq_flow"
		stages: [
			#JOINT_OFFSET_FLOW,
			//#MATCH_OFFSETS_FLOW,
			//#MISD_TMP_FLOW,
			#RELAX_FLOW,
			#POST_ALIGN_FLOW,
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
