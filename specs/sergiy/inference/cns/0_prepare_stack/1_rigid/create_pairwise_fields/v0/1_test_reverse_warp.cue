#FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field_inv"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3375]
	end_coord: [2048, 2048, 3380]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

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
		//on_info_exists:      "overwrite"
		write_procs: _ | *[]
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x55"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     100
batch_gap_sleep_sec: 0.05
local_test:          true
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#FLOW_TMPL & {
			src: path: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/resin_mask"
			dst: path: "gs://sergiy_exp/aced/tmp/rigid_to_elastic/resin_mask_inv"
			dst_resolution: [256, 256, 45]
			mode: "mask"
		},
	]
}
