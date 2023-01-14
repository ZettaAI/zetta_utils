#Z_OFFSET: -1
#TGT_OFFSET: [0, 0, #Z_OFFSET]

#Z_ADJUSTER: {
	"@type": "VolumetricIndexTranslator"
	offset: [0, 0, #Z_OFFSET]
	resolution: [4, 4, 30]
}

#FOLDER:        "large_test_x8"
#IMG_PATH:      "gs://zfish_unaligned/coarse_x0/raw_masked"
#BASE_ENC_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x0"
#ENC_PATH:      "gs://zfish_unaligned/coarse_x0/encodings_masked"

#FIELD_FWD_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/field_\(#Z_OFFSET)_fwd"
#FIELD_BWD_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/field_\(#Z_OFFSET)_bwd"

#IMG_WARPED_PATH:      "gs://sergiy_exp/aced/zfish/\(#FOLDER)/img_warped_\(#Z_OFFSET)"
#WARPED_BASE_ENC_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/base_enc_warped_\(#Z_OFFSET)"

#BASE_ENC_MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"

#MISD_MODEL_PATH: "gs://sergiy_exp/training_artifacts/aced_misd/zm1_zm2_thr1.0_scratch_large_custom_dset_x2/checkpoints/epoch=2-step=1524.ckpt.static-1.12.1+cu102-model.jit"

//#MISD_MODEL_PATH: "gs://sergiy_exp/training_artifacts/aced_misd/thr1.1_x4/last.ckpt.static-1.12.1+cu102-model.jit"
//#MISD_MODEL_PATH: "gs://sergiy_exp/training_artifacts/aced_misd/thr2.1_x0/last.ckpt.static-1.12.1+cu102-model.jit"
#MISD_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/misd_\(#Z_OFFSET)_with_zeros_v3"

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 3002]
	end_coord: [1024, 1024, 3012]
	resolution: [512, 512, 30]
}

#STAGES: [
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
			"data_type":    "float32"
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
			"data_type":    "float32"
		}
		on_info_exists: "overwrite"
	}
}

#CF_FWD_FLOW: #CF_FLOW_TMPL & {
	dst: path: #FIELD_FWD_PATH
	tmp_layer_dir: "\(#FIELD_FWD_PATH)/tmp"
	tgt_offset: [0, 0, #Z_OFFSET]
}

#CF_BWD_FLOW: #CF_FLOW_TMPL & {
	dst: path: #FIELD_BWD_PATH
	tmp_layer_dir: "\(#FIELD_BWD_PATH)/tmp"
	src_offset: [0, 0, #Z_OFFSET]
}

#INV_FLOW: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "invert_field"
			"@mode": "partial"
		}
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [1024, 1024, 1]
	}
	idx: {
		"@type": "VolumetricIndex"
		bcube:   #BCUBE
		resolution: [32, 32, 30]
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_FWD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #FIELD_BWD_PATH
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "overwrite"
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_warp_flow"
	mode:    "img"
	crop: [256, 256, 0]
	chunk_size: [2048, 2048, 1]
	bcube:          #BCUBE
	dst_resolution: #STAGES[len(#STAGES)-1].dst_resolution
	src: {
		"@type": "build_cv_layer"
		path:    _
		index_adjs: [#Z_ADJUSTER]
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_BWD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: _
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

#IMG_WARP_FLOW: #WARP_FLOW_TMPL & {
	src: path:                #IMG_PATH
	dst: path:                #IMG_WARPED_PATH
	dst: info_reference_path: #IMG_PATH
}

#ENCODE_FLOW: {
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
		path:    #IMG_WARPED_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #WARPED_BASE_ENC_PATH
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

#MISD_FLOW: {
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
		path:    #WARPED_BASE_ENC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #MISD_PATH
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x20"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     15
worker_lease_sec:    10
batch_gap_sleep_sec: 3

local_test: false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				//#CF_FWD_FLOW,
				//#CF_BWD_FLOW,
			]
		},
		//#IMG_WARP_FLOW,
		//#ENCODE_FLOW,
		#MISD_FLOW,
	]
}
