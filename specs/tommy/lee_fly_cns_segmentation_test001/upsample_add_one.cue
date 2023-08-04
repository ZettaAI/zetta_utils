//
// Handy variables
#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_mask_stage1_v2_64nm_try_x1_iter300_rig200_lr0.001_3consecutive"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_mask_stage1_v2_64nm_try_x1_iter300_rig200_lr0.001_3consecutive_semantic_v2"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [1024 * 20, 8 * 1024, 2500]
	end_coord: [38 * 1024, 12 * 1024, 3500]
	resolution: [16, 16, 45]
}

// Execution parameters
"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x189"
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
	dst_resolution: [16, 16, 45]

	// How do we chunk/crop/blend?
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1]]
	processing_crop_pads: [[0, 0, 0]]
	processing_blend_pads: [[0, 0, 0]]

	// We want to expand the input bbox to be evenly divisible
	// by chunk size
	expand_bbox_processing: true

	// Specification for the operation we're performing
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src + 2"
	}
	// Specification for the inputs to the operation
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
			data_resolution: [64, 64, 45]
			interpolation_mode: "mask"
		}
	}

	// Specification of the output layer. Subchunkable expects
	// a single output layer. If multiple output layers are
	// needed, refer to advanced examples.
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_field_overrides: {"type": "segmentation"}
		on_info_exists: "overwrite"
	}
}
