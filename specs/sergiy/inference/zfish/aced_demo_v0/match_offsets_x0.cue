#ENC_PATH: "gs://zfish_unaligned/coarse_x0/encodings_masked"

#FOLDER:        "large_test_x8"
#MISD_ZM1_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/misd_-1_with_zeros_v3"
#MISD_ZM2_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/misd_-2_with_zeros_v3"

#MATCH_OFFSETS_PATH: "gs://sergiy_exp/aced/zfish/\(#FOLDER)/match_offsets_with_zeros_v3"

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 3002]
	end_coord: [1024, 1024, 3012]
	resolution: [512, 512, 30]
}

#MATCH_OFFSET_FLOW: {
	"@type": "build_get_match_offsets_flow"
	bcube:   #BCUBE
	chunk_size: [2048, 2048, 1]
	dst_resolution: [32, 32, 30]
	non_tissue: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
		read_postprocs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "=="
				value:   0
			},
		]
	}
	misd_mask_zm1: {
		"@type": "build_cv_layer"
		path:    #MISD_ZM1_PATH
	}
	misd_mask_zm2: {
		"@type": "build_cv_layer"
		path:    #MISD_ZM2_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #MATCH_OFFSETS_PATH
		info_reference_path: #MISD_ZM1_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x15"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     15
worker_lease_sec:    10
batch_gap_sleep_sec: 3

local_test: false

target: #MATCH_OFFSET_FLOW
