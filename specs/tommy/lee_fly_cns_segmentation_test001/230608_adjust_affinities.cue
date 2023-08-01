#AFF_PATH:    "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity"
#BACKUP_PATH: "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity/backup_for_adjustments"

// #AFF_PATH:           "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity/cutouts/test001"
// #BACKUP_PATH:        "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity/cutouts/test001/backup"
#BLACKOUT_MSK_PATH:  "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/resin_mask_final_surgery"
#SNAP_MSK_PATH:      "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_mask_stage1_v2_64nm_try_x1_iter300_rig200_lr0.001_surgery_consecutive3"
#THRESHOLD_MSK_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/aff_mask_stage1_v2_64nm_try_x1_iter300_rig200_lr0.001_surgery"

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:tmacrina_mask_affinities_x3"
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas: 512
local_test:      false

#AFF: {
	"@type": "build_cv_layer"
	path:    #AFF_PATH
	cv_kwargs: {"delete_black_uploads": false} // ws+agg require all affinities present, so cannot issue DELETES
}
#BACKUP: {
	"@type":             "build_cv_layer"
	path:                #BACKUP_PATH
	info_reference_path: #AFF_PATH
}
#BLACKOUT_MASK: {
	"@type": "build_cv_layer"
	path:    #BLACKOUT_MSK_PATH
	data_resolution: [256, 256, 45]
	interpolation_mode: "mask"
}
#SNAP_MASK: {
	"@type": "build_cv_layer"
	path:    #SNAP_MSK_PATH
	data_resolution: [64, 64, 45]
	interpolation_mode: "mask"
}
#THRESHOLD_MASK: {
	"@type": "build_cv_layer"
	path:    #THRESHOLD_MSK_PATH
	data_resolution: [64, 64, 45]
	interpolation_mode: "mask"
}

#BBOX: {
	"@type": "BBox3D.from_coords"
	// start_coord: [1341 - 144, 288, 2990]
	// end_coord: [1341, 288 + 144, 2990 + 2*13]
	// resolution: [256, 256, 45]
	start_coord: [0, 0, 0]
	// end_coord: [4096, 4608, 7010] // actual
	end_coord: [4176, 4752, 7020] // divisible by processing_chunk_size
	resolution: [256, 256, 45]
}

target: {
	"@type": "build_subchunkable_apply_flow"
	// "@mode": "partial"
	bbox: #BBOX
	dst:  #AFF
	dst_resolution: [16, 16, 45]
	processing_chunk_sizes: [[144 * 16, 144 * 16, 13]]
	// must be one pixel for lowest-res mask in xy and >=1 in z for snap mask breaking
	processing_crop_pads: [[16, 16, 1]]
	processing_blend_pads: [[0, 0, 0]]
	op: {
		"@type": "AdjustAffinitiesOp"
	}
	op_kwargs: {
		aff_layer:            #AFF
		aff_backup_layer:     #BACKUP
		blackout_mask_layer:  #BLACKOUT_MASK
		snap_mask_layer:      #SNAP_MASK
		threshold_mask_layer: #THRESHOLD_MASK
		threshold_value:      0.85
		fill_value:           0
	}
}
