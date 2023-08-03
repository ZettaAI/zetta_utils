#IMG_PATH:     "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1/raw_img"
#BASE_FOLDER:  "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0"
#MED_IMG_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_aligned_1024nm_try_x6_iter8000_rig0.5_lr0.005"
#TMP_PATH:     "gs://tmp_2s/yo/"

#AFIELD_NAME:     "afield_1024nm_try_x6_iter8000_rig0.5_lr0.005"
#AFIELD_PATH:     "\(#BASE_FOLDER)/\(#AFIELD_NAME)"
#AFIELD_INV_PATH: "\(#BASE_FOLDER)/\(#AFIELD_NAME)_inv"

#PFIELDS_DIR:     "\(#BASE_FOLDER)/fields_fwd"
#PFIELDS_INV_DIR: "\(#BASE_FOLDER)/fields_inv"

#C2M_FIELD_DIR:     "\(#BASE_FOLDER)/\(#AFIELD_NAME)_C2M_field"
#C2M_IMG_DIR:       "\(#BASE_FOLDER)/\(#AFIELD_NAME)_C2M_img"
#C2M_FIELD_INV_DIR: "\(#BASE_FOLDER)/\(#AFIELD_NAME)_C2M_field_inv"
#C2M_IMG_INV_DIR:   "\(#BASE_FOLDER)/\(#AFIELD_NAME)_C2M_img_inv"

#M2M_FIELD_DIR:     "\(#BASE_FOLDER)/\(#AFIELD_NAME)_M2M_field_fwd"
#M2M_FIELD_INV_DIR: "\(#BASE_FOLDER)/\(#AFIELD_NAME)_M2M_field_inv"
#M2M_IMG_DIR:       "\(#BASE_FOLDER)/\(#AFIELD_NAME)_M2M_img"
#M2M_IMG_INV_DIR:   "\(#BASE_FOLDER)/\(#AFIELD_NAME)_M2M_img_inv"

#AFIELD_RESOLUTION: [1024, 1024, 45]
#PFIELD_RESOLUTION: [32, 32, 45]
#RFIELD_RESOLUTION: [64, 64, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 799]
	end_coord: [2048, 2048, 811]
	resolution: [512, 512, 45]
}

#INVERT_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial", mode: "torchfields"}
	expand_bbox_processing: true

	processing_chunk_sizes: [[512, 512, 1]]
	//max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [64, 64, 0]
	//level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

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

#WARP_FLOW_TMPL: {
	"@type":                   "build_subchunkable_apply_flow"
	processing_crop_pads:      _ | *null
	processing_chunk_sizes:    _
	level_intermediaries_dirs: _ | *null
	dst_resolution:            _
	op: {
		"@type":                 "WarpOperation"
		translation_granularity: _ | *1
		mode:                    _
	}
	bbox: #BBOX
	src: {
		"@type":            "build_ts_layer"
		path:               _
		index_procs:        _ | *[]
		data_resolution:    _ | *null
		interpolation_mode: _ | *null
	}
	field: {
		"@type":            "build_ts_layer"
		path:               _
		index_procs:        _ | *[]
		data_resolution:    _ | *null
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		index_procs:         _ | *[]
		info_reference_path: _ | *src.path
	}
}

#EXTRACT_C2M_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #RFIELD_RESOLUTION
	op: mode: "field"

	src: path: "\(#PFIELDS_DIR)/\(_z_offset)"

	field: path:            #AFIELD_PATH
	field: data_resolution: #AFIELD_RESOLUTION
	field: index_procs: [
		{
			"@type": "VolumetricIndexTranslator"
			offset: [0, 0, _z_offset]
			resolution: [4, 4, 45]
		},
	]
	op: translation_granularity: field.data_resolution[0] / dst_resolution[0]

	dst: path: "\(#C2M_FIELD_DIR)/\(_z_offset)"
	_z_offset: int
}

#TEST_C2M_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [32, 32, 45]
	op: mode:  "img"
	src: path: #IMG_PATH

	field: path:                 "\(#C2M_FIELD_DIR)/\(_z_offset)"
	field: data_resolution:      #RFIELD_RESOLUTION
	op: translation_granularity: field.data_resolution[0] / dst_resolution[0]

	dst: path: "\(#C2M_IMG_DIR)/\(_z_offset)"
	_z_offset: int
}

#EXTRACT_C2M_INV_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #RFIELD_RESOLUTION
	op: mode: "field"

	field: path: "\(#PFIELDS_INV_DIR)/\(_z_offset)"

	src: path:               #AFIELD_INV_PATH
	src: data_resolution:    #AFIELD_RESOLUTION
	src: interpolation_mode: "field"
	src: index_procs: [
		{
			"@type": "VolumetricIndexTranslator"
			offset: [0, 0, _z_offset]
			resolution: [4, 4, 45]
		},
	]
	op: translation_granularity: src.data_resolution[0] / dst_resolution[0]

	dst: path: "\(#C2M_FIELD_INV_DIR)/\(_z_offset)"
	_z_offset: int
}

#TEST_C2M_INV_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [32, 32, 45]
	op: mode:  "img"
	src: path: #MED_IMG_PATH
	src: index_procs: [
		{
			"@type": "VolumetricIndexTranslator"
			offset: [0, 0, _z_offset]
			resolution: [4, 4, 45]
		},
	]
	field: path:                 "\(#C2M_FIELD_INV_DIR)/\(_z_offset)"
	field: data_resolution:      #RFIELD_RESOLUTION
	op: translation_granularity: field.data_resolution[0] / dst_resolution[0]

	dst: path: "\(#C2M_IMG_INV_DIR)/\(_z_offset)"
	_z_offset: int
}

#EXTRACT_M2M_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #RFIELD_RESOLUTION
	op: mode:                    "field"
	op: translation_granularity: src.data_resolution[0] / dst_resolution[0]

	src: path:               #AFIELD_INV_PATH
	src: data_resolution:    #AFIELD_RESOLUTION
	src: interpolation_mode: "field"

	field: path: "\(#C2M_FIELD_DIR)/\(_z_offset)"

	dst: path: "\(#M2M_FIELD_DIR)/\(_z_offset)"
	_z_offset: int
}

#TEST_M2M_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [32, 32, 45]
	op: mode:  "img"
	src: path: #MED_IMG_PATH

	field: path:            "\(#M2M_FIELD_DIR)/\(_z_offset)"
	field: data_resolution: #RFIELD_RESOLUTION

	dst: path: "\(#M2M_IMG_DIR)/\(_z_offset)"
	_z_offset: int
}

#EXTRACT_M2M_INV_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #RFIELD_RESOLUTION
	op: mode:                    "field"
	op: translation_granularity: field.data_resolution[0] / dst_resolution[0]

	field: path:               #AFIELD_PATH
	field: data_resolution:    #AFIELD_RESOLUTION
	field: interpolation_mode: "field"

	src: path: "\(#C2M_FIELD_INV_DIR)/\(_z_offset)"

	dst: path: "\(#M2M_FIELD_INV_DIR)/\(_z_offset)"
	_z_offset: int
}

#TEST_M2M_INV_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [32, 32, 45]
	op: mode:  "img"
	src: path: #MED_IMG_PATH
	src: index_procs: [
		{
			"@type": "VolumetricIndexTranslator"
			offset: [0, 0, _z_offset]
			resolution: [4, 4, 45]
		},
	]

	field: path:            "\(#M2M_FIELD_INV_DIR)/\(_z_offset)"
	field: data_resolution: #RFIELD_RESOLUTION

	dst: path: "\(#M2M_IMG_INV_DIR)/\(_z_offset)"
	_z_offset: int
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x155"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_replicas:        200
batch_gap_sleep_sec:    1
local_test:             false
#Z_OFFSETS: [-1, -2]
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		//#INVERT_FLOW,
		for offset in #Z_OFFSETS {
			"@type": "mazepa.seq_flow"
			stages: [
				{
					"@type": "mazepa.concurrent_flow"
					stages: [
						#EXTRACT_C2M_FLOW & {_z_offset:     offset},
						#EXTRACT_C2M_INV_FLOW & {_z_offset: offset},
					]
				},
				{
					"@type": "mazepa.concurrent_flow"
					stages: [
						#EXTRACT_M2M_FLOW & {_z_offset:     offset},
						#EXTRACT_M2M_INV_FLOW & {_z_offset: offset},
					]
				},
				{
					"@type": "mazepa.concurrent_flow"
					stages: [
						#TEST_M2M_FLOW & {_z_offset:     offset},
						#TEST_M2M_INV_FLOW & {_z_offset: offset},
					]
				},

			]
		},
	]
}
