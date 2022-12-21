#SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
#DST_PATH: "gs://tmp_2w/inference_tests/remote/raw_img_x0_tommy"

#FLOW_SPEC: {
	"@type": "build_write_flow"
	chunk_size: [1024, 1024, 1]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 7, 1024 * 3, 2000]
		end_coord: [1024 * 9, 1024 * 5, 2200]
		resolution: [64, 64, 40]
	}
	dst_resolution: [64, 64, 40]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "expect_same"
	}
}

"@type": "mazepa.execute_on_gcp_with_sqs"
target:  #FLOW_SPEC

worker_image: "us.gcr.io/zetta-research/zetta_utils:inference_x11"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     45
worker_lease_sec:    5
batch_gap_sleep_sec: 0.5

local_test: false
