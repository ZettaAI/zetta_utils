#BASE_FOLDER:  "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x0/production/v1"
#FIRST_FIELD:  "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/field"
#SECOND_FIELD: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x0/try_x1/afield_try_x4_iter500_rig20"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 429]
	end_coord: [2048, 2048, 975]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "field"
	}
	expand_bbox: true
	processing_chunk_sizes: [[#CHUNK, #CHUNK, 1]]
	processing_crop_pads: [[#CROP_PAD, #CROP_PAD, 0]]
	//level_intermediaries_dirs: ["file://~/.zutils/tmp", "file://~/.zutils/tmp"]
	dst_resolution: [256, 256, 45]
	bbox: #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    #FIRST_FIELD
	}
	field: {
		"@type": "build_ts_layer"
		path:    "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x0/try_x1/afield_try_x4_iter500_rig20"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#BASE_FOLDER)/field"
		info_reference_path: #SECOND_FIELD
		//on_info_exists:      "overwrite"
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x131"
worker_resources: {
	memory: "18560Mi"
}
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_replicas:        100
batch_gap_sleep_sec:    0.1
local_test:             false
target:                 #FLOW
