// RUN ACED BLOCK

// INPUTS
#COARSE_FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"

//#IMG_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"

#IMG_PATH:      "gs://zfish_unaligned/coarse_x0/raw_img_masked"
#BASE_ENC_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x0"
#ENC_PATH:      "gs://zfish_unaligned/coarse_x0/encodings_masked"

#COARSE_IMG_PATH: "gs://zfish_unaligned/coarse_x0/raw_img"

// MODELS
#BASE_ENC_MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"
#MISD_MODEL_PATH:     "gs://sergiy_exp/training_artifacts/aced_misd/zm1_zm2_thr1.0_scratch_large_custom_dset_x2/checkpoints/epoch=2-step=1524.ckpt.static-1.12.1+cu102-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "cutout_g_x10"

#FOLDER:          "gs://sergiy_exp/aced/zfish/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

#MATCH_OFFSETS_PATH: "\(#FOLDER)/debug_x0/match_offsets"

#AFIELD_PATH:             "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH:        "\(#FOLDER)/img_aligned\(#RELAXATION_SUFFIX)"
#IMG_MASK_PATH:           "\(#FOLDER)/img_mask\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_MASKED_PATH: "\(#FOLDER)/img_aligned_masked\(#RELAXATION_SUFFIX)"

#CF_INFO_CHUNK: [512, 512, 1]
#AFIELD_INFO_CHUNK: [512, 512, 1]
#RELAXATION_CHUNK: [512, 512, #Z_END - #Z_START]
#RELAXATION_FIX:  "first"
#RELAXATION_ITER: 200

#RELAXATION_RIG: 300
//#COMPUTE FIELD DONE FOR 4 -> 25

#Z_START: 2957

#Z_END: 2959
//#Z_START: 3058
//#Z_END:   3098

#RELAXATION_SUFFIX: "_try_x0" //"_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [256 + 128, 256 + 128, #Z_END]
	resolution: [512, 512, 30]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [50000, 60000, #Z_START + 1]
	scale_bar_nm: 30000
	layers: [
		["input", "image", "precomputed://\(#IMG_PATH)"],
		["-1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/-1"],
		["-2 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/-2"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
	]
}

#NOT_FIRST_SECTION_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START + 1]
	end_coord: [256 + 128, 256 + 128, #Z_END]
	resolution: [512, 512, 30]
}
#FIRST_SECTION_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [256 + 128, 256 + 128, #Z_START + 1]
	resolution: [512, 512, 30]
}

#STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 30]

		fn: {
			sm:       250
			num_iter: 500
			lr:       0.015
		}
		chunk_size: [512, 512, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 30]
		fn: {
			sm:       100
			num_iter: 400
			lr:       0.015
		}
		chunk_size: [1024, 1024, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [128, 128, 30]
		fn: {
			sm:       100
			num_iter: 400
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [128, 128, 30]
		fn: {
			sm:       50
			num_iter: 300
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [64, 64, 30]
		fn: {
			sm:       20
			num_iter: 150
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [32, 32, 30]
		fn: {
			sm:       15
			num_iter: 75
		}
		chunk_size: [2048, 2048, 1]
	},

]
#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	chunk_size:     _
	src?:           _
	tgt?:           _
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
	"@type":     "build_compute_field_multistage_flow"
	bbox:        #NOT_FIRST_SECTION_BBOX
	stages:      #STAGES
	src_offset?: _
	tgt_offset?: _
	offset_resolution: [4, 4, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		info_chunk_size:     #CF_INFO_CHUNK
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
		info_reference_path: dst.info_reference_path
		info_chunk_size:     dst.info_chunk_size
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
			encoding:     "zfpc"
		}
		on_info_exists: "overwrite"
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_warp_flow"
	mode:    _
	crop_pad: [256, 256, 0]
	chunk_size: [2048, 2048, 1]
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
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		write_procs?:   _
		index_procs?:   _ | *[]
	}
}

#ENCODE_FLOW_TMPL: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":     "BaseEncoder"
			model_path:  #BASE_ENC_MODEL_PATH
			uint_output: true
		}
		crop_pad: [32, 32, 0]
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [2048, 2048, 1]
	}
	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BBOX
		resolution: [32, 32, 30]
	}
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		info_field_overrides: {
			"data_type": "uint8"
		}
	}
}

#MISD_FLOW_TMPL: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "MisalignmentDetector"
			model_path: #MISD_MODEL_PATH
		}
		crop_pad: [32, 32, 0]
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [2048, 2048, 1]
	}
	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BBOX
		resolution: [32, 32, 30]
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #BASE_ENC_PATH
	}
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		info_field_overrides: {
			"data_type": "uint8"
		}
	}
}

#INVERT_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[64, 64, 0]]
	level_intermediaries_dirs: ["file://~/.zutils/cache/"]
	dst_resolution: [32, 32, 30]
	bbox: #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type": "build_volumetric_layer_set"
		layers: {
			yo: {
				"@type":             "build_cv_layer"
				path:                _
				info_reference_path: src.path
				on_info_exists:      "overwrite"
			}
		}
	}
}

#Z_OFFSETS: [-1, -2]
#MATCH_OFFSETS_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX
	op: {
		"@type": "AcedMatchOffsetOp"
	}
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[128, 128, 0]]
	level_intermediaries_dirs: ["file://~/.zutils/cache/"]

	dst_resolution: [32, 32, 30]

	tissue_mask: {
		"@type": "build_cv_layer"
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
	misd_masks: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_cv_layer"
				path:    "\(#MISALIGNMENTS_PATH)/\(offset)"
			}
		}
	}
	pairwise_fields: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_cv_layer"
				path:    "\(#FIELDS_INV_PATH)/\(offset)"
			}
		}
	}
	pairwise_fields_inv: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_cv_layer"
				path:    "\(#FIELDS_PATH)/\(offset)"
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
				info_chunk_size: [1024, 1024, 1]
				on_info_exists: "overwrite"
			}
		}

	}
}

#RELAX_FLOW: {
	"@type":         "build_aced_relaxation_flow"
	fix:             #RELAXATION_FIX
	num_iter:        #RELAXATION_ITER
	rigidity_weight: #RELAXATION_RIG

	bbox:       #BBOX
	chunk_size: #RELAXATION_CHUNK
	crop_pad: [96, 96, 0]
	dst_resolution: [32, 32, 30]
	match_offsets: {
		"@type": "build_cv_layer"
		path:    #MATCH_OFFSETS_PATH
	}
	field_zm1: {
		"@type": "build_cv_layer"
		path:    "\(#FIELDS_PATH)/-1"
	}
	field_zm2: {
		"@type": "build_cv_layer"
		path:    "\(#FIELDS_PATH)/-2"
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

#APPLY_MASK_FLOW_TMPL: {
	"@type": "build_apply_mask_flow"
	chunk_size: [2048, 2048, 1]
	dst_resolution: [32, 32, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #IMG_ALIGNED_PATH
	}
	masks: [
		{
			"@type":     "build_cv_layer"
			path:        _
			read_procs?: _ | *[]
		},
	]
	dst: {
		"@type":             "build_cv_layer"
		path:                #IMG_ALIGNED_MASKED_PATH
		info_reference_path: #IMG_PATH
		on_info_exists:      "Overwrite"
	}
	bbox: _
}

#JOINT_POST_ALIGN_FLOW: {
	"@type": "mazepa.sequential_flow"
	stages: [
		{
			"@type": "mazepa.sequential_flow"
			stages: [
				{
					"@type": "mazepa.concurrent_flow"
					stages: [
						#WARP_FLOW_TMPL & {
							mode: "img"
							src: path:   #IMG_PATH
							field: path: #AFIELD_PATH
							dst: path:   #IMG_ALIGNED_PATH
						},
						#WARP_FLOW_TMPL & {
							mode: "mask"
							src: path: #MATCH_OFFSETS_PATH
							src: read_procs: [
								{
									"@type": "compare"
									"@mode": "partial"
									mode:    "=="
									value:   0
								},
							]
							field: path: #AFIELD_PATH
							dst: path:   #IMG_MASK_PATH
							dst: write_procs: [
								{
									"@type": "to_uint8"
									"@mode": "partial"
								},
							]
						},
					]
				},
				#APPLY_MASK_FLOW_TMPL & {
					masks: [{path: #IMG_MASK_PATH}]
					bbox: #NOT_FIRST_SECTION_BBOX
				},
			]
		},
		{
			#APPLY_MASK_FLOW_TMPL & {
				bbox: #FIRST_SECTION_BBOX
				masks: [
					{
						path: #BASE_ENC_PATH
						read_procs: [
							{
								"@type": "compare"
								"@mode": "partial"
								mode:    "=="
								value:   127
							},
							{
								"@type": "filter_cc"
								"@mode": "partial"
								mode:    "keep_large"
								thr:     1000
							},

						]
					},
				]
			}
		},
	]
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x91"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      40
	batch_gap_sleep_sec:  1
	do_dryrun_estimation: true
	local_test:           true

	target: {
		"@type": "mazepa.sequential_flow"
		stages: [
			//#JOINT_OFFSET_FLOW,
			//#MATCH_OFFSETS_FLOW,
			#INVERT_FLOW_TMPL & {
				src: path: "\(#FIELDS_PATH)/-1"
				dst: layers: yo: path: "\(#FIELDS_INV_PATH)/tmp/-1_set_yo"
			}

			//#RELAX_FLOW,
			//#JOINT_POST_ALIGN_FLOW,,,,,,,,,,,,,,,,,,
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
