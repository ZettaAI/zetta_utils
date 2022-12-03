#SRC_PATH: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
#DST_PATH: "gs://tmp_2w/inference_tests/remote/raw_img_x0"

"@type": "mazepa.execute"
exec_queue: {
	"@type":            "mazepa.SQSExecutionQueue"
	name:               "sergiy-corgie-x0"
	outcome_queue_name: "sergiy-crogie-completion-x0"
	pull_lease_sec:     30
}
target: {
	"@type": "build_write_flow"
	chunk_size: [1024, 1024, 1]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 7, 1024 * 3, 2000]
		end_coord: [1024 * 9, 1024 * 5, 2005]
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
