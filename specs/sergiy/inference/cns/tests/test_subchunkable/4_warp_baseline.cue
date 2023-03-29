#FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2000]
	end_coord: [1024, 1024, 2001]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024

#FLOW_TMPL: {
	"@type": "build_warp_flow"
	crop_pad: [#CROP_PAD, #CROP_PAD, 0]
	chunk_size: [#CHUNK, #CHUNK, 1]
	dst_resolution: _
	mode:           _
	bbox:           #BBOX
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
			dst: path: "gs://tmp_2w/baseline/defect_mask"
			dst_resolution: [64, 64, 45]
			dst: write_procs: [
				{"@type": "to_uint8", "@mode": "partial"},
			]
			mode: "mask"
		},
		#FLOW_TMPL & {
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid"
			dst: path: "gs://tmp_2w/baseline/raw_img"
			dst_resolution: [32, 32, 45]
			mode: "img"
		},

	]
}
