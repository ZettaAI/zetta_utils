#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1"
#MASKS_PATH:  "\(#BASE_FOLDER)/defect_mask"
#SRC_PATH:    "\(#BASE_FOLDER)/encodings"
#DST_PATH:    "\(#BASE_FOLDER)/encodings_masked"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 6150]
	end_coord: [2048, 2048, 6170]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type":                "build_subchunkable_apply_flow"
	expand_bbox_processing:            true
	processing_chunk_sizes: _
	dst_resolution:         _
	fn: {
		"@type": "apply_mask_fn"
		"@mode": "partial"
	}
	bbox: #BBOX
	src: {
		"@type":    "build_ts_layer"
		path:       _
		read_procs: _ | *[]
	}
	masks: [
		{
			"@type":    "build_ts_layer"
			path:       #MASKS_PATH
			read_procs: _ | *[]
		},
	]
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x139"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:        20
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
batch_gap_sleep_sec:    1.0
local_test:             false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in [32, 64, 128, 256, 512, 1024, 2048, 4096] {
			#FLOW_TMPL & {
				src: path: #SRC_PATH
				dst: path: #DST_PATH
				dst_resolution: [res, res, 45]
				processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1]]
			}
		},

	]
}
