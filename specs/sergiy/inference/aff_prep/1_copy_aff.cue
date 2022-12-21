#SRC_PATH: "gs://zetta_lee_mouse_spinal_cord_001_image/dorsal_sections/dorsal_sections_500/affinity/minnie_mye_v2"
#DST_PATH: "gs://sergiy_exp/aff_dsets/x0/aff"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:inference_x11"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     15
worker_lease_sec:    15
batch_gap_sleep_sec: 5
local_test:          false
target: {
	"@type": "build_write_flow"
	chunk_size: [512, 512, 1]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 200, 1024 * 80, 200]
		end_coord: [1024 * 205, 1024 * 85, 400]
		resolution: [4, 4, 45]
	}
	dst_resolution: [8, 8, 45]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_chunk_size: [128, 128, 1]
		on_info_exists: "overwrite"
	}
}
