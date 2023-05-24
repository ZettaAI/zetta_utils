#FIELD_PATH:  "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 1000]
	end_coord: [2048, 2048, 2000]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	processing_chunk_sizes: [[2 * #CHUNK, 2 * #CHUNK, 1], [#CHUNK, #CHUNK, 1]]
	processing_crop_pads: [[0, 0, 0], [#CROP_PAD, #CROP_PAD, 0]]
	expand_bbox_processing:    true
	dst_resolution: _
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
		info_reference_path: _
		//on_info_exists:      "overwrite"
		write_procs: _ | *[]
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x112"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     100
batch_gap_sleep_sec: 0.1
local_test:          false
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
