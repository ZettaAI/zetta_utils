// RUN ACED BLOCK

// INPUTS
#COARSE_FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#IMG_PATH:          "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
#BASE_ENC_PATH:     "gs://zfish_unaligned/precoarse_x0/base_enc_x0"
#ENC_PATH:          "gs://zfish_unaligned/precoarse_x0/encodings_masked"
#COARSE_IMG_PATH:   "gs://zfish_unaligned/coarse_x0/raw_img"

// MODELS
#BASE_ENC_MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"
#MISD_MODEL_PATH:     "gs://sergiy_exp/training_artifacts/aced_misd/zm1_zm2_thr1.0_scratch_large_custom_dset_x2/checkpoints/epoch=2-step=1524.ckpt.static-1.12.1+cu102-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "coarse_no_iter_x0"
#FOLDER:          "gs://sergiy_exp/aced/zfish/\(#PAIRWISE_SUFFIX)"
#FIELDS_FWD_PATH: "\(#FOLDER)/fields_fwd"
#FIELDS_BWD_PATH: "\(#FOLDER)/fields_bwd"

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
#RELAXATION_CHUNK: [512, 512, 4]
#RELAXATION_FIX:  "both"
#RELAXATION_ITER: 150
#RELAXATION_RIG:  20

#Z_START:           0
#Z_END:             3
#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, #Z_START]
	end_coord: [1024, 1024, #Z_END]
	resolution: [512, 512, 30]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [50000, 60000, #Z_START + 1]
	scale_bar_nm: 30000
	layers: [
		["precoarse_img", "image", "precomputed://\(#IMG_PATH)"],
		["coarse_img", "image", "precomputed://\(#COARSE_IMG_PATH)"],
		["-1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/-1"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
	]
}

#NOT_FIRST_SECTION_BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, #Z_START + 1]
	end_coord: [1024, 1024, #Z_END]
	resolution: [512, 512, 30]
}
#FIRST_SECTION_BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, #Z_START]
	end_coord: [1024, 1024, #Z_START + 1]
	resolution: [512, 512, 30]
}

#STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 30]

		operation: fn: {
			sm:       400
			num_iter: 2
			lr:       0.015
		}
		chunk_size: [1024, 1024, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 30]

		operation: fn: {
			sm:       200
			num_iter: 0
			lr:       0.015
		}
		chunk_size: [1024, 1024, 1]
	},
]

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	chunk_size:     _
	operation: {
		"@type": "ComputeFieldOperation"
		fn: {
			"@type":  "align_with_online_finetuner"
			"@mode":  "partial"
			sm:       _
			num_iter: _
			lr?:      _
		}
		crop_pad: [64, 64, 0]
	}
}

#CF_FLOW_TMPL: {
	"@type":     "build_compute_field_multistage_flow"
	bcube:       #BCUBE
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
	src_field: {
		"@type": "build_cv_layer"
		path:    #COARSE_FIELD_PATH
		data_resolution: [256, 256, 30]
		interpolation_mode: "field"
	}
	tgt_field: {
		"@type": "build_cv_layer"
		path:    #COARSE_FIELD_PATH
		data_resolution: [256, 256, 30]
		interpolation_mode: "field"
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
		on_info_exists: "expect_same"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
			encoding:     "zfpc"
		}
		on_info_exists: "expect_same"
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_warp_flow"
	mode:    _
	crop: [256, 256, 0]
	chunk_size: [2048, 2048, 1]
	bcube:          #BCUBE
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	src: {
		"@type":         "build_cv_layer"
		path:            _
		read_postprocs?: _
		index_adjs?:     _ | *[]
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
		on_info_exists:  "expect_same"
		write_preprocs?: _
		index_adjs?:     _ | *[]
	}
}

#ENCODE_FLOW_TMPL: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseEncoder"
			model_path: #BASE_ENC_MODEL_PATH
		}
		crop: [32, 32, 0]
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [2048, 2048, 1]
	}
	idx: {
		"@type": "VolumetricIndex"
		bcube:   #BCUBE
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
		crop: [32, 32, 0]
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [2048, 2048, 1]
	}
	idx: {
		"@type": "VolumetricIndex"
		bcube:   #BCUBE
		resolution: [32, 32, 30]
	}
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
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

#WARP_FWD_FLOW: #WARP_FLOW_TMPL & {
	mode: "img"
	src: path: #IMG_PATH
	dst: index_adjs: [
		{
			"@type": "VolumetricIndexTranslator"
			offset: [0, 0, -1]
			resolution: [4, 4, 30]
		},
	]
	field: path: "\(#FIELDS_FWD_PATH)/-1"
	dst: path:   "\(#IMGS_WARPED_PATH)/+1"
}
#Z_OFFSETS: [-1]
#JOINT_OFFSET_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				//#CF_FLOW_TMPL & {
				// dst: path: "\(#FIELDS_FWD_PATH)/\(z_offset)"
				//tmp_layer_dir: "\(#FIELDS_FWD_PATH)/\(z_offset)/tmp"
				//tgt_offset: [0, 0, z_offset]
				//},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						#CF_FLOW_TMPL & {
							dst: path: "\(#FIELDS_BWD_PATH)/\(z_offset)"
							tmp_layer_dir: "\(#FIELDS_BWD_PATH)/\(z_offset)/tmp"
							src_offset: [0, 0, z_offset]
						},
						#WARP_FLOW_TMPL & {
							mode: "img"
							src: path: #IMG_PATH
							src: index_adjs: [
								{
									"@type": "VolumetricIndexTranslator"
									offset: [0, 0, z_offset]
									resolution: [4, 4, 30]
								},
							]
							field: path: "\(#FIELDS_BWD_PATH)/\(z_offset)"
							dst: path:   "\(#IMGS_WARPED_PATH)/\(z_offset)"
						},
					]
				},
			]
		},
	]
}

#RUN_INFERENCE: {
	"@type":        "mazepa.execute_on_gcp_with_sqs"
	max_task_retry: 33
	worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x52"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:     20
	worker_lease_sec:    20
	batch_gap_sleep_sec: 1

	local_test: true

	target: {
		"@type": "mazepa.seq_flow"
		stages: [
			#JOINT_OFFSET_FLOW,
			//#MATCH_OFFSETS_FLOW,
			//#RELAX_FLOW,
			//#JOINT_POST_ALIGN_FLOW,
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
