import "math"

import "list"

#TMP_PATH: "gs://tmp_2w/prepare_cns"

#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1"

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [int, int, int] | *[32768, 36864, 7015]
	// end_coord: [32768, 32768, 3001]
	resolution: [32, 32, 45]
}

#GET_TISSUE_MASK_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[2 * 1024, 2 * 1024, 1]]
	processing_crop_pads: [[0, 1024, 1024]]
	expand_bbox: true
	dst_resolution: [32, 32, 45]
	bbox: #ROI_BOUNDS

	src: {
		"@type": "build_cv_layer"
		path:    "\(#BASE_FOLDER)/encodings"
	}

	dst: {
		"@type":             "build_cv_layer"
		info_reference_path: "\(#BASE_FOLDER)/raw_img"
		path:                "\(#BASE_FOLDER)/tissue_mask"
		//on_info_exists:      "overwrite"
		write_procs: [
			{"@type": "compare", "@mode":        "partial", value: 0, mode:     "=="},
			{"@type": "filter_cc", "@mode":      "partial", thr:   12500, mode: "keep_large"},
			{"@type": "binary_closing", "@mode": "partial", width: 3},
			{"@type": "erode", "@mode":          "partial", width: 15},
			{"@type": "compare", "@mode":        "partial", value: 0, mode:  "=="},
			{"@type": "filter_cc", "@mode":      "partial", thr:   80, mode: "keep_large"},
			{"@type": "to_uint8", "@mode":       "partial"},
		]
	}
}

#EXECUTE_TMPL: {
	"@type":                "mazepa.execute_on_gcp_with_sqs"
	worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x190"
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"
	worker_replicas:        int
	worker_resources:       _
	local_test:             false
	target:                 _
}
#EXECUTE_ON_CPU: #EXECUTE_TMPL & {
	worker_resources: {
		memory: "18560Mi"
	}
	worker_replicas: 200
}

#EXECUTE_ON_GPU: #EXECUTE_TMPL & {
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas: 100
}

#GET_TISSUE_MASK: #EXECUTE_ON_CPU & {
	target: #GET_TISSUE_MASK_FLOW
}

[
	#GET_TISSUE_MASK,
]
