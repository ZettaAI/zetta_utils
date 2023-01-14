#FOLDER: "large_test_x8"

#IMG_PATH:           "gs://zfish_unaligned/coarse_x0/raw_masked"
#FIELD_ZM1_PATH:     "gs://sergiy_exp/aced/zfish/\(#FOLDER)/field_-1_fwd"
#FIELD_ZM2_PATH:     "gs://sergiy_exp/aced/zfish/\(#FOLDER)/field_-2_fwd"
#MATCH_OFFSETS_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/match_offsets_with_zeros_v3"

#VERSION:         "debug_x7_both"
#AFIELD_PATH:     "gs://sergiy_exp/aced/zfish/\(#FOLDER)/afield_\(#VERSION)"
#IMG_WARPED_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/final_img_\(#VERSION)"

#BCUBE: {
	"@type": "BoundingCube"
	//start_coord: [0, 0, 3002]
	//end_coord: [1024, 1024, 3012]
	start_coord: [512, 512, 3002]
	end_coord: [512 + 64, 512 + 64, 3012]
	resolution: [512, 512, 30]
}

#RELAX_FLOW: {
	"@type": "build_aced_relaxation_flow"
	bcube:   #BCUBE
	chunk_size: [1024, 1024, 10]
	crop: [64, 64, 0]
	dst_resolution: [32, 32, 30]
	match_offsets: {
		"@type": "build_cv_layer"
		path:    #MATCH_OFFSETS_PATH
	}
	field_zm1: {
		"@type": "build_cv_layer"
		path:    #FIELD_ZM1_PATH
	}
	field_zm2: {
		"@type": "build_cv_layer"
		path:    #FIELD_ZM2_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFIELD_PATH
		info_reference_path: #FIELD_ZM1_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
	fix: "both"
	//num_iter: 150
}

#WARP_FLOW: {
	"@type": "build_warp_flow"
	mode:    "img"
	bcube:   #BCUBE
	crop: [256, 256, 0]
	chunk_size: [1024, 1024, 1]
	dst_resolution: [32, 32, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #IMG_PATH
	}
	field: {
		"@type": "build_cv_layer"
		path:    #AFIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #IMG_WARPED_PATH
		info_reference_path: #IMG_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "expect_same"
	}
}

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x21"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     15
worker_lease_sec:    90
batch_gap_sleep_sec: 5

local_test: true

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#RELAX_FLOW,
		#WARP_FLOW,
	]
}
