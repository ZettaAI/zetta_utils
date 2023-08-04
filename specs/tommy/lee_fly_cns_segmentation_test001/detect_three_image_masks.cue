//
// Handy variables
#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_mask_stage1_v2_64nm_try_x1_iter300_rig200_lr0.001"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_mask_stage1_v2_64nm_try_x1_iter300_rig200_lr0.001_3consecutive_v2"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [5120, 1024, 6000]
	end_coord: [7168, 3072, 6011]
	resolution: [64, 64, 45]
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
local_test:      true // set to `false` execute remotely

target: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX

	dst_resolution: [64, 64, 45]

	processing_chunk_sizes: [[1024, 1024, 12]]
	// need to manually set crop/pad for consecutive detection
	processing_crop_pads: [[0, 0, 2]]
	processing_blend_pads: [[0, 0, 0]]

	expand_bbox_processing: true

	// Specification for the operation we're performing
	fn: {
		"@type": "detect_consecutive_masks"
		"@mode": "partial"
	}
	// Specification for the inputs to the operation
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
		num_consecutive: 3
	}

	// Specification of the output layer. Subchunkable expects
	// a single output layer. If multiple output layers are
	// needed, refer to advanced examples.
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
	}
}
