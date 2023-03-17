#R2E_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#E2R_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field_inv"
#R2R_DIR:  "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_rigid/v1/fields"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3400]
	end_coord: [2048, 2048, 3500]
	resolution: [512, 512, 45]
}
#CROP_PAD: 512
#CHUNK:    1024 * 2

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	processing_crop_pads: [[#CROP_PAD, #CROP_PAD, 0]]
	processing_chunk_sizes: [[#CHUNK, #CHUNK, 1]]
	dst_resolution: [256, 256, 45]
	op: {
		"@type": "WarpOperation"
		mode:    "field"
	}
	bbox: #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    #R2E_PATH
	}
	field: {
		"@type": "build_cv_layer"
		path:    #E2R_PATH
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
		path:                "\(#R2R_DIR)/\(_z_offset)"
		info_reference_path: src.path
		//on_info_exists:      "overwrite"
	}
	_z_offset: int
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x112"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     10
batch_gap_sleep_sec: 0.05
local_test:          false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for offset in [-1, -2] {
			#FLOW_TMPL & {_z_offset: offset}
		},
	]
}
