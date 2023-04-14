#FIELD_DIR: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_rigid/v1/fields"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [2048, 2048, 1000]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	processing_crop_pads: [[#CROP_PAD, #CROP_PAD, 0]]
	processing_chunk_sizes: [[#CHUNK, #CHUNK, 1]]
	dst_resolution: _
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	bbox: #BBOX
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
		path:    _
		data_resolution: [256, 256, 45]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		//on_info_exists:      "overwrite"
		write_procs: _ | *[]
		index_procs: _ | *[]
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x129"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     200
batch_gap_sleep_sec: 0.05
local_test:          true

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in [-3] {
			#FLOW_TMPL & {
				src: path:   "gs://zetta_lee_fly_cns_001_alignment_temp/cns/rigid_x0/raw_img"
				dst: path:   "gs://sergiy_exp/aced/tmp/r2r/img_x0/\(z_offset)"
				field: path: "\(#FIELD_DIR)/\(z_offset)"
				dst_resolution: [256, 256, 45]
				dst: index_procs: [
					{
						"@type": "VolumetricIndexTranslator"
						offset: [0, 0, z_offset]
						resolution: [4, 4, 45]
					},
				]
				op: mode: "img"
			}
		},
	]
}
