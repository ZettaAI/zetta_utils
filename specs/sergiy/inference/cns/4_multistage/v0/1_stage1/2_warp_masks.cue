#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced"

#IMG_PATH: "\(#BASE_FOLDER)/coarse_x1/raw_img"

#ENC_PATH: "\(#BASE_FOLDER)/coarse_x1/encodings_masked"
#TMP_PATH: "gs://tmp_2w/temporary_layers"

#PAIRWISE_SUFFIX: "try_x0"

#FOLDER: "\(#BASE_FOLDER)/med_x1/\(#PAIRWISE_SUFFIX)"

#STAGE0_DEFECTS_PATH:       "\(#BASE_FOLDER)/coarse_x1/defect_mask"
#STAGE0_MISALIGNMENTS_PATH: "\(#FOLDER)/misalignments"
#STAGE0_TISSUE_MASK_PATH:   "\(#BASE_FOLDER)/tissue_mask"

#STAGE1_DEFECTS_PATH:       "\(#FOLDER)/defect_mask_stage1\(#STAGE0_RELAXATION_SUFFIX)"
#STAGE1_MISALIGNMENTS_PATH: "\(#FOLDER)/misalignments_stage1\(#STAGE0_RELAXATION_SUFFIX)"

#STAGE1_TISSUE_MASK_PATH: "\(#FOLDER)/tissue_mask_stage1\(#STAGE0_RELAXATION_SUFFIX)"

//#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
//#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

//#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
//#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"

#STAGE0_RELAXATION_SUFFIX: "_1024nm_try_x14_iter8000_rig0.5_lr0.001_clip0.01"

#AFIELD_PATH: "\(#FOLDER)/afield\(#STAGE0_RELAXATION_SUFFIX)"

#AFIELD_RESOLUTION: [1024, 1024, 45]
//#MATCH_OFFSETS_BASE: "\(#FOLDER)/match_offsets_z\(#Z_START)_\(#Z_END)"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [2048, 2048, 2250]
	resolution: [512, 512, 45]
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
		write_procs:         _ | *[]
	}
}

#WARP_MISALIGNMENTS_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[8 * 1024, 8 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [64, 64, 45]
	op: mode: "img"

	src: path: "\(#STAGE0_MISALIGNMENTS_PATH)/\(_z_offset)"

	field: path:            #AFIELD_PATH
	field: data_resolution: #AFIELD_RESOLUTION

	dst: path:                "\(#STAGE1_MISALIGNMENTS_PATH)/\(_z_offset)"
	dst: info_reference_path: src.path
	_z_offset: int
}

#WARP_TISSUE_MASK_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[8 * 1024, 8 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [64, 64, 45]
	op: mode: "img"

	src: path: #STAGE0_TISSUE_MASK_PATH

	field: path:            #AFIELD_PATH
	field: data_resolution: #AFIELD_RESOLUTION

	dst: path:                #STAGE1_TISSUE_MASK_PATH
	dst: info_reference_path: src.path
	dst: write_procs: [
		{
			"@type": "compare"
			"@mode": "partial"
			mode:    ">"
			value:   0.4
		},
		{
			"@type": "to_uint8"
			"@mode": "partial"
		},
	]
}

#WARP_DEFECT_MASK_FLOW: #WARP_FLOW_TMPL & {
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	processing_chunk_sizes: [[8 * 1024, 8 * 1024, 1], [2 * 1024, 2 * 1024, 1]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [64, 64, 45]
	op: mode: "img"

	src: path: #STAGE0_DEFECTS_PATH

	field: path:            #AFIELD_PATH
	field: data_resolution: #AFIELD_RESOLUTION

	dst: path:                #STAGE1_DEFECTS_PATH
	dst: info_reference_path: src.path
	dst: write_procs: [
		{
			"@type": "compare"
			"@mode": "partial"
			mode:    ">"
			value:   0.4
		},
		{
			"@type": "to_uint8"
			"@mode": "partial"
		},
	]
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x184"
worker_resources: {
	memory: "18560Mi"
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
		#WARP_DEFECT_MASK_FLOW,
		#WARP_TISSUE_MASK_FLOW,

		for offset in #Z_OFFSETS {
			#WARP_MISALIGNMENTS_FLOW & {_z_offset: offset}
		},
	]
}
