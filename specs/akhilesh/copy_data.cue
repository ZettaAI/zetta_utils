//
// Handy variables
#SRC_PATH: "gs://dacey-human-retina-001-segmentation/240320-retina-finetune-embd24-v1-x1/20240328/ws"
#DST_PATH: "gs://zetta_ws/dacey_human_fovea_2404_x0"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [5 * 1024, 5 * 1024, 1025]
	end_coord: [6 * 1024, 6 * 1024, 1025 + 128]
	resolution: [20, 20, 50]
}

// Execution parameters
"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:akhilesh_copy_x0"
worker_cluster_name:    "zutils-x3"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas: 10
local_test:      false // set to `false` execute remotely

target: {
	// We're applying subchunkable processing flow,
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX

	// What resolution is our destination?
	dst_resolution: [20, 20, 50]

	// How do we chunk/crop/blend?
	processing_chunk_sizes: [[512, 512, 64]]
	processing_crop_pads: [[0, 0, 0]]
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
