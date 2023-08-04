#SRC_PATH:    "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [2048, 2048, 8000]
	resolution: [512, 512, 45]
}

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[2 * 1024, 2 * 1024, 1]]
	processing_crop_pads: [[0, 0, 0]]
	expand_bbox_processing: true
	dst_resolution: [256, 256, 45]
	bbox: #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#BASE_FOLDER)/field"
		info_reference_path: #SRC_PATH
		//on_info_exists:      "overwrite"
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x112"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     50
batch_gap_sleep_sec: 0.1
local_test:          false
target:              #FLOW
