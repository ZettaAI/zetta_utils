// RUN ACED BLOCK

// INPUTS
#COARSE_FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#IMG_PATH:          "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
#BASE_ENC_PATH:     "gs://zfish_unaligned/coarse_x0/base_enc_x0"

//#ENC_PATH: "gs://zfish_unaligned/precoarse_x0/encodings_masked"
#ENC_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"

//OUTPUTS
#FOLDER:      "gs://sergiy_exp/aced/zfish/pairs_test_x1/zutils_m6_pad64_nokeys_x0_no_tgt_sm"
#FIELDS_PATH: "\(#FOLDER)/fields"

#IMGS_WARPED_PATH: "\(#FOLDER)/imgs_warped"

#CF_INFO_CHUNK: [512, 512, 1]

#Z_START: 1
#Z_END:   3

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_END]
	resolution: [512, 512, 30]
}
#NOT_FIRST_SECTION_BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, #Z_START + 1]
	end_coord: [2048, 2048, #Z_END]
	resolution: [512, 512, 30]
}
#FIRST_SECTION_BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_START + 1]
	resolution: [512, 512, 30]
}

#STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 30]

		operation: fn: {
			sm:       200
			num_iter: 500
			lr:       0.015
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 30]

		operation: fn: {
			sm:       200
			num_iter: 500
			lr:       0.015
		}
		chunk_size: [2048, 2048, 1]
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
	bcube:       #NOT_FIRST_SECTION_BCUBE
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
	bcube: #BCUBE
	dst_resolution: [256, 256, 30]
	src: {
		"@type":         "build_cv_layer"
		path:            _
		read_postprocs?: _
		index_adjs?:     _ | *[]
	}
	field: {
		"@type":            "build_cv_layer"
		data_resolution:    #STAGES[len(#STAGES)-1].dst_resolution
		interpolation_mode: "field"
		path:               _
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

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 3
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x57"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     10
worker_lease_sec:    20
batch_gap_sleep_sec: 1

local_test:           true
do_dryrun_estimation: true
show_progress:        true

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#CF_FLOW_TMPL & {
			dst: path: "\(#FIELDS_PATH)/field"
			tmp_layer_dir: "\(#FIELDS_PATH)/tmp"
			tgt_offset: [0, 0, -1]
		},
		#WARP_FLOW_TMPL & {
			mode: "img"
			src: path:   #IMG_PATH
			field: path: "\(#FIELDS_PATH)/field"
			dst: path:   "\(#IMGS_WARPED_PATH)/img"
		},
	]
}
