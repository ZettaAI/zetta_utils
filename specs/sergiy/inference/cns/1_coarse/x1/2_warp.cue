#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1"
#TMP_PATH:    "gs://tmp_2w/temporary_layers"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 430]
	end_coord: [2048, 2048, 440]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [512, 512, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: _
	bbox:           #BBOX
	src: {
		"@type":    "build_ts_layer"
		path:       _
		read_procs: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
		path:    "\(#BASE_FOLDER)/field"
		data_resolution: [256, 256, 45]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: _
		//on_info_exists:      "overwrite"
		write_procs: _ | *[]
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x139"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:        50
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
batch_gap_sleep_sec:    1.0
local_test:             false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#FLOW_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/resin/ResinNet20221115_29k"
			src: read_procs: [
				{"@type": "compare", "@mode": "partial", mode: ">=", value: 48},
			]
			dst: path:                "\(#BASE_FOLDER)/resin_mask"
			dst: info_reference_path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/resin_mask"
			dst: write_procs: [
				{"@type": "to_uint8", "@mode": "partial"},
			]
			dst_resolution: [256, 256, 45]
			op: mode: "mask"
		},
		#FLOW_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/defects/DefectNet20221114_50k"
			src: read_procs: [
				{"@type": "compare", "@mode": "partial", mode: ">=", value: 48},
			]
			dst: path: "\(#BASE_FOLDER)/defect_mask"
			dst_resolution: [64, 64, 45]
			dst: info_reference_path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
			dst: write_procs: [
				{"@type": "to_uint8", "@mode": "partial"},
			]
			op: mode: "mask"
		},
		#FLOW_TMPL & {
			src: path:                "gs://zetta_lee_fly_cns_001_alignment_temp/rigid"
			dst: path:                "\(#BASE_FOLDER)/raw_img"
			dst: info_reference_path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img"
			dst_resolution: [32, 32, 45]
			op: mode: "img"
		},

	]
}
