#FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"

#Z_START: 2000
#Z_END:   2001
#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	processing_crop_pads: [[0, 0, 0], [512, 512, 0]]
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024, 1024, 1]]
	max_reduction_chunk_sizes: [1024 * 4, 1024 * 4, 1]
	dst_resolution: _
	start_coord: [0, 0, #Z_START]
	end_coord: [1024, 1024, #Z_END]
	coord_resolution: [512, 512, 45]
	temp_layers_dirs: ["gs://tmp_2w/tmp_dirs", "file://~/.cloudvolume/memcache/"]
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
		data_resolution: [256, 256, 45]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		write_procs:    _ | *[]
	}
	allow_cache_up_to_level: 1
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x57"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     3
batch_gap_sleep_sec: 0.05
local_test:          true
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#FLOW_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/defects/DefectNet20221114_50k"
			src: read_procs: [
				{"@type": "compare", "@mode": "partial", mode: ">=", value: 48},
			]
			dst: path: "gs://tmp_2w/subch/defect_mask"
			dst_resolution: [64, 64, 45]
			dst: write_procs: [
				{"@type": "to_uint8", "@mode": "partial"},
			]
			op: mode: "mask"
		},
		#FLOW_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid"
			dst: path: "gs://tmp_2w/subch/raw_img"
			dst_resolution: [32, 32, 45]
			op: mode: "img"
		},

	]
}
