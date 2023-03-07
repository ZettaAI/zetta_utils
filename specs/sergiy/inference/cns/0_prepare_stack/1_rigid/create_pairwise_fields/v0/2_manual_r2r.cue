#R2E_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#E2R_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field_inv"
#R2R_DIR:  "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_rigid/v1/fields"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3370]
	end_coord: [2048, 2048, 3380]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

#FLOW_TMPL: {
	"@type": "build_warp_flow"
	crop_pad: [#CROP_PAD, #CROP_PAD, 0]
	chunk_size: [#CHUNK, #CHUNK, 1]
	dst_resolution: [256, 256, 45]
	mode: "mask"
	bbox: #BBOX
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
		path:    _
		index_procs: [
			{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, _z_offset]
				resolution: [4, 4, 45]
			},
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		//on_info_exists:      "overwrite"
	}
	_z_offset: int | *0
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
			src: path: "gs://zetta_lee_fly_cns_001_alignment_temp/resin/ResinNet20221115_29k"
			src: read_procs: [
				{"@type": "compare", "@mode":  "partial", mode: ">=", value: 48},
				{"@type": "to_uint8", "@mode": "partial"},
			]
			field: path: #R2E_PATH
			dst: path:   "gs://sergiy_exp/aced/tmp/rigid_to_elastic/resin_mask_r2e"
			_z_offset: 0
		},
		#FLOW_TMPL & {
			src: path:   "gs://sergiy_exp/aced/tmp/rigid_to_elastic/resin_mask_r2e"
			field: path: #E2R_PATH
			dst: path:   "gs://sergiy_exp/aced/tmp/rigid_to_elastic/resin_mask_r2e_e2r_zm1"
			_z_offset: -1
		},
	]
}
