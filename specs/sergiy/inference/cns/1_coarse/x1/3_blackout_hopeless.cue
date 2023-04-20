#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1"

#SRC_PATH: "\(#BASE_FOLDER)/img_raw"
#FLOW: {
	"@type": "build_annotated_section_copy_flow"
	start_coord_xy: [0, 0]
	end_coord_xy: [2048, 2048]
	coord_resolution_xy: [512, 512]
	chunk_size_xy: [2048, 2048]
	fill_resolutions: [
		[32, 32, 45],
	]

	annotation_path: "cns_x0/seethrough_section_candidates_x0"
	src: {
		"@type":             "build_ts_layer"
		path:                "file://~/.zutils/tmp"
		info_reference_path: #SRC_PATH
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		cv_kwargs: {
			"delete_black_uploads": false
		}
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x139"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:        30
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"

batch_gap_sleep_sec: 1.0
local_test:          false

target: #FLOW
