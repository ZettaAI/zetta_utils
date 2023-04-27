#IMG_PATH:     "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1/raw_img"
#BASE_FOLDER:  "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0"
#MED_IMG_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_aligned_1024nm_try_x6_iter8000_rig0.5_lr0.005"
#TMP_PATH:     "gs://tmp_2s/yo/"

#AFIELD_NAME:     "afield_1024nm_try_x14_iter8000_rig0.5_lr0.001_clip0.01"
#AFIELD_PATH:     "\(#BASE_FOLDER)/\(#AFIELD_NAME)"
#AFIELD_INV_PATH: "\(#BASE_FOLDER)/\(#AFIELD_NAME)_inv"

#AFIELD_RESOLUTION: [1024, 1024, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2250]
	end_coord: [2048, 2048, 3155]
	resolution: [512, 512, 45]
}

#INVERT_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial", mode: "torchfields"}
	expand_bbox: true

	processing_chunk_sizes: [[512, 512, 1], [64, 64, 1]]
	max_reduction_chunk_sizes: [512, 512, 1]
	processing_crop_pads: [[0, 0, 0], [24, 24, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #AFIELD_RESOLUTION
	bbox:           #BBOX
	src: {
		"@type": "build_ts_layer"
		path:    #AFIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFIELD_INV_PATH
		info_reference_path: src.path
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x184"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_replicas:        20
batch_gap_sleep_sec:    1
local_test:             false

target: #INVERT_FLOW
