#SRC_PATH: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"

#FLOW: {
	"@type": "build_annotated_section_copy_flow"
	start_coord_xy: [0, 0]
	end_coord_xy: [2048, 2048]
	coord_resolution_xy: [512, 512]
	chunk_size_xy: [2048, 2048]
	fill_resolutions: [
		for res in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
			[res, res, 45]
		},
	]

	annotation_path: "cns_x0/seethrough_section_candidates_x0"
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		index_procs: [
			{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, -1]
				resolution: [4, 4, 45]
			},
		]
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x55"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     30
batch_gap_sleep_sec: 0.05

local_test: false

target: #FLOW
