#FOLDER: "large_test_x8"

#IMG_PATH: "gs://zfish_unaligned/coarse_x0/raw_masked"

#VERSION:                "debug_x6"
#AFIELD_PATH:            "gs://sergiy_exp/aced/zfish/\(#FOLDER)/afield_\(#VERSION)"
#IMG_WARPED_PATH:        "gs://sergiy_exp/aced/zfish/\(#FOLDER)/final_img_\(#VERSION)"
#MATCH_OFFSETS_PATH:     "gs://sergiy_exp/aced/zfish/\(#FOLDER)/match_offsets_with_zeros_v3"
#IMG_MASK_WARPED_PATH:   "gs://sergiy_exp/aced/zfish/\(#FOLDER)/final_img_mask_\(#VERSION)"
#IMG_WARPED_MASKED_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/final_img_\(#VERSION)_masked"

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 3002]
	end_coord: [1024, 1024, 3012]
	//start_coord: [128, 512, 3002]
	//end_coord: [128 + 64, 512 + 64, 3012]
	resolution: [512, 512, 30]
}

#WARP_MASK_FLOW: {
	"@type": "build_warp_flow"
	mode:    "img"
	bcube:   #BCUBE
	crop: [256, 256, 0]
	chunk_size: [1024, 1024, 1]
	dst_resolution: [32, 32, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #MATCH_OFFSETS_PATH
		read_postprocs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "=="
				value:   0
			},
		]
	}
	field: {
		"@type": "build_cv_layer"
		path:    #AFIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #IMG_MASK_WARPED_PATH
		info_reference_path: #MATCH_OFFSETS_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		write_preprocs: [
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
}
#MASK_FLOW: {
	"@type": "build_apply_mask_flow"
	chunk_size: [2048, 2048, 1]
	dst_resolution: [32, 32, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #IMG_WARPED_PATH
	}
	mask: {
		"@type": "build_cv_layer"
		path:    #IMG_MASK_WARPED_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #IMG_WARPED_MASKED_PATH
		info_reference_path: #IMG_WARPED_PATH
	}
	bcube: #BCUBE
}

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x13"
worker_resources: {
	memory: "18560Mi"
	//      "nvidia.com/gpu": "1"
}
worker_replicas:     15
worker_lease_sec:    90
batch_gap_sleep_sec: 5

local_test: false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#WARP_MASK_FLOW,
		#MASK_FLOW,
	]
}
