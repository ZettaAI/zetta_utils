//
// Handy variables
#SRC_PATH: "gs://dacey-human-retina-001-segmentation/240320-retina-finetune-embd24-v1-x1/20240408/seg_agg25"
#DST_PATH: "gs://dodam_exp/seg_medcutoutlod"
#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [1024 * 0, 1024 * 0, 1995]
	end_coord: [1024 * 10, 1024 * 10, 1995 + 128]
	resolution: [20, 20, 50]
}

// Execution parameters
"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p310_x214"
worker_cluster_name:    "zutils-x3"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas: 10

local_test:      true // set to `false` execute remotely

target: {
	// We're applying subchunkable processing flow,
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX

	// What resolution is our destination?
	dst_resolution: [20, 20, 50]

	// How do we chunk/crop/blend?
	processing_chunk_sizes: [[2 * 1024, 2 * 1024, 64]]
	processing_crop_pads: [[1, 0, 0]]
	processing_blend_pads: [[0, 0, 0]]
	skip_intermediaries: true

	// We want to expand the input bbox to be evenly divisible
	// by chunk size
	expand_bbox_processing: true

	// Specification for the operation we're performing
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	// Specification for the inputs to the operation
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}

	// Specification of the output layer. Subchunkable expects
	// a single output layer. If multiple output layers are
	// needed, refer to advanced examples.
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		//on_info_exists:      "overwrite"
	}
}
