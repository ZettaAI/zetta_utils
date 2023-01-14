// RUN ACED BLOCK 

// INPUTS
#IMG_PATH:      "gs://zfish_unaligned/coarse_x0/raw_masked"
#BASE_ENC_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x0"
#ENC_PATH:      "gs://zfish_unaligned/coarse_x0/encodings_masked"

// MODELS
#BASE_ENC_MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"
#MISD_MODEL_PATH:     "gs://sergiy_exp/training_artifacts/aced_misd/zm1_zm2_thr1.0_scratch_large_custom_dset_x2/checkpoints/epoch=2-step=1524.ckpt.static-1.12.1+cu102-model.jit"

//OUTPUTS
#FOLDER: "gs://sergiy_exp/aced/zfish/joint_test_x0"

#PAIR_SUFFIX:     "_debug_x5"
#FIELDS_FWD_PATH: "\(#FOLDER)/fields_fwd\(#PAIR_SUFFIX)"
#FIELDS_BWD_PATH: "\(#FOLDER)/fields_bwd\(#PAIR_SUFFIX)"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped\(#PAIR_SUFFIX)"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped\(#PAIR_SUFFIX)"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments\(#PAIR_SUFFIX)"

#MATCH_OFFSETS_PATH: "\(#FOLDER)/match_offsets"

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 1]
	end_coord: [1024, 1024, 2]
	resolution: [512, 512, 30]
}

#STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 30]

		operation: fn: {
			sm:       100
			num_iter: 250
		}
		chunk_size: [1024, 1024, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 30]

		operation: fn: {
			sm:       100
			num_iter: 250
		}
		chunk_size: [1024, 1024, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [128, 128, 30]

		operation: fn: {
			sm:       100
			num_iter: 300
		}
		chunk_size: [1024, 1024, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [64, 64, 30]

		operation: fn: {
			sm:       25
			num_iter: 150
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [32, 32, 30]

		operation: fn: {
			sm:       25
			num_iter: 75
		}
		chunk_size: [2048, 2048, 1]
	},
]

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	crop:           64
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":  "align_with_online_finetuner"
			"@mode":  "partial"
			sm:       _
			num_iter: _
		}
		crop: [128, 128, 0]
	}
	chunk_size: _
}

#CF_FLOW_TMPL: {
	"@type":     "build_compute_field_multistage_flow"
	bcube:       #BCUBE
	stages:      #STAGES
	src_offset?: _
	tgt_offset?: _
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
		info_chunk_size: [1024, 1024, 1]
		info_field_overrides: {
			"num_channels": 2
			encoding:       "zfpc"
			data_type:      "float32"
		}
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		info_field_overrides: {
			"num_channels": 2
			encoding:       "zfpc"
			"data_type":    "float32"
		}
		on_info_exists: "overwrite"
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

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x30"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     8
worker_lease_sec:    10
batch_gap_sleep_sec: 3

local_test: false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#JOINT_OFFSET_FLOW,
		//#MATCH_OFFSETS_FLOW,
		//#RELAX_FLOW,
		//#JOINT_POST_ALIGN_FLOW,
	]
}
