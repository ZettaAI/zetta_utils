//
// Handy variables
#SRC_PATH: "gs://zheng_mouse_hippocampus_scratch_30/img_aligned_try_384nm_iter8000_rig3.0_lr0.001_cutout001"
#DST_PATH: "gs://sergiy_exp/hippo/missing/cutout_x0/img"

#MIDDLE: 1266
#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [36 * 1024, 42 * 1024, #MIDDLE - 16]
	end_coord: [37 * 1024, 43 * 1024, #MIDDLE + 16]
	resolution: [24, 24, 45]
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
debug: true
local_test:      true // set to `false` execute remotely

target: {
	// We're applying subchunkable processing flow,
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX

	// What resolution is our destination?
	dst_resolution: [24, 24, 45]

	// How do we chunk/crop/blend?
	processing_chunk_sizes: [[1024, 1024, 16]]
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
		on_info_exists:      "overwrite"
        info_add_scales: [[24, 24, 45]]
        // info_add_scales_ref: {
        //    "chunk_sizes": [[128, 128, 1]], 
        //    "encoding": "raw", 
        //    "sharding": null,
        //    "resolution": [12, 12, 45], 
        //    "size": [131072, 131072, 3388], 
        //    "voxel_offset": [0, 0, 0]
        // }
        info_chunk_size: [128, 128, 1]
        info_add_scales_exclude_fields: ["sharding"]
        info_add_scales_mode: "replace"
	}
}
