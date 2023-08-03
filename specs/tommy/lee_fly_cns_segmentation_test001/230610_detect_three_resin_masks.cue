// Create a "snap" mask at locations of three consecutive image masks

#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/resin_mask_final_surgery"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/resin_mask_final_surgery_consecutive3"

#BBOX: {
	"@type": "BBox3D.from_coords"
	// start_coord: [9216, 0, 4000]
	// end_coord: [10240, 1024, 4064]
	start_coord: [0, 0, 0]
	end_coord: [4096, 4608, 7010]
	resolution: [256, 256, 45]
}

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:tmacrina_mask_affinities_x3"
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas: 32
local_test:      false

target: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX
	dst_resolution: [256, 256, 45]
	processing_chunk_sizes: [[2048, 2048, 64]]
	// need to manually set crop/pad for consecutive detection
	processing_crop_pads: [[0, 0, 2]]
	processing_blend_pads: [[0, 0, 0]]

	expand_bbox_processing: true

	fn: {
		"@type": "detect_consecutive_masks"
		"@mode": "partial"
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
		num_consecutive: 3
	}

	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
	}
}
