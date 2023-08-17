		// INPUTS
#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced"
#TMP_PATH:    "gs://tmp_2w/temporary_layers"

#STAGE0_RELAXATION_SUFFIX: "_1024nm_try_x14_iter8000_rig0.5_lr0.001_clip0.01"

#STAGE1_DEFECTS_PATH:       "\(#FOLDER)/defect_mask_stage1\(#STAGE0_RELAXATION_SUFFIX)"
#STAGE1_MISALIGNMENTS_PATH: "\(#FOLDER)/misalignments_stage1\(#STAGE0_RELAXATION_SUFFIX)"
#STAGE1_TISSUE_MASK_PATH:   "\(#FOLDER)/tissue_mask_stage1\(#STAGE0_RELAXATION_SUFFIX)"
#IMG_PATH:                  "\(#FOLDER)/img_aligned\(#STAGE0_RELAXATION_SUFFIX)"

#MATCH_OFFSET_BASE: "\(#FOLDER)/match_offsets_stage1_v1"

//OUTPUTS
#PAIRWISE_SUFFIX: "try_x0"

#FOLDER: "\(#BASE_FOLDER)/med_x1/\(#PAIRWISE_SUFFIX)"

//#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
//#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#FIELDS_PATH:     "\(#FOLDER)/afield\(#STAGE0_RELAXATION_SUFFIX)_M2M_field_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/afield\(#STAGE0_RELAXATION_SUFFIX)_M2M_field_inv"

#AFIELD_PATH:      "\(#FOLDER)/afield_stage1\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH: "\(#FOLDER)/img_aligned_stage1\(#RELAXATION_SUFFIX)_x0"
#IMG_MASK_PATH:    "\(#FOLDER)/img_mask_stage1\(#RELAXATION_SUFFIX)"
#AFF_MASK_PATH:    "\(#FOLDER)/aff_mask_stage1\(#RELAXATION_SUFFIX)"

#RELAX_OUTCOME_CHUNK: [256, 256, 1]
#RELAXATION_FIX:  "both"
#RELAXATION_ITER: 300
#RELAXATION_LR:   1e-3

#RELAXATION_RIG: 200

//#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"
#DEBUG_SUFFIX:      ""
#RELAXATION_SUFFIX: "_64nm_try_x1_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_lr\(#RELAXATION_LR)\(#DEBUG_SUFFIX)"
#RELAXATION_RESOLUTION: [64, 64, 45]
#BLOCKS: [
	//{_z_start: 0, _z_end: 151},
	// {_z_start: 150, _z_end:  301},
	// {_z_start: 300, _z_end:  452},
	// {_z_start: 451, _z_end:  600},
	// {_z_start: 599, _z_end:  755},
	// {_z_start: 754, _z_end:  901},
	// {_z_start: 900, _z_end:  1053},
	// {_z_start: 1052, _z_end: 1207},
	// {_z_start: 1206, _z_end: 1350},
	// {_z_start: 1349, _z_end: 1502},
	// {_z_start: 1501, _z_end: 1653},
	// {_z_start: 1652, _z_end: 1803},
	// {_z_start: 1802, _z_end: 1951},
	// {_z_start: 1950, _z_end: 2104},
	// {_z_start: 2103, _z_end: 2251},
	//{_z_start: 2250, _z_end: 2401},
	//{_z_start: 2400, _z_end: 2551},
	//{_z_start: 2550, _z_end: 2702},
	//{_z_start: 2701, _z_end: 2851},
	//{_z_start: 2850, _z_end: 3003},
	//{_z_start: 3002, _z_end: 3155},
	// ^^^ Proofread
	//{_z_start: 3154, _z_end: 3302},
	{_z_start: 3301, _z_end: 3450},
	{_z_start: 3449, _z_end: 3600},
	//         {_z_start: 3599, _z_end: 3751},
	//         {_z_start: 3750, _z_end: 3895},
	//         {_z_start: 3894, _z_end: 4051},
	//         {_z_start: 4050, _z_end: 4196},
	//         {_z_start: 4195, _z_end: 4348},
	//         {_z_start: 4347, _z_end: 4501},
	//         {_z_start: 4500, _z_end: 4652},
	//         {_z_start: 4651, _z_end: 4805},
	//         {_z_start: 4804, _z_end: 4953},
	//         {_z_start: 4952, _z_end: 5103},
	//         {_z_start: 5102, _z_end: 5254},
	//         {_z_start: 5253, _z_end: 5403},
	//         {_z_start: 5402, _z_end: 5555},
	//         {_z_start: 5554, _z_end: 5701},
	//         {_z_start: 5700, _z_end: 5853},
	//         {_z_start: 5852, _z_end: 6002},
	//         {_z_start: 6001, _z_end: 6152},
	//         {_z_start: 6151, _z_end: 6301},
	//         {_z_start: 6300, _z_end: 6451},
	//         {_z_start: 6450, _z_end: 6601},
	//         {_z_start: 6600, _z_end: 6751},
	//         {_z_start: 6750, _z_end: 6901},
	//         {_z_start: 6900, _z_end: 7050},

]

#BBOX_TMPL: {
	"@type":  "BBox3D.from_coords"
	_z_start: int
	_z_end:   int
	start_coord: [0, 0, _z_start]
	end_coord: [36864, 36864, _z_end]
	resolution: [32, 32, 45]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [50000, 60000, 1000]
	scale_bar_nm: 30000
	layers: [
		["precoarse_img", "image", "precomputed://\(#IMG_PATH)"],
		//["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX) -2", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
	]
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	//chunk_size: [512, 512, 1]
	bbox:           _
	dst_resolution: _ | *[32, 32, 45]
	src: {
		"@type":      "build_ts_layer"
		path:         _
		read_procs?:  _
		index_procs?: _ | *[]
	}
	field: {
		"@type":            "build_ts_layer"
		path:               _
		data_resolution:    _ | *null
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		write_procs?:        _
		index_procs?:        _ | *[]
	}
}

#Z_OFFSETS: [-1, -2]
#MATCH_OFFSETS_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "AcedMatchOffsetOp"
	}
	bbox: _

	processing_chunk_sizes: [[512, 512, bbox._z_end - bbox._z_start]]
	//max_reduction_chunk_sizes: [1024, 1024, bbox._z_end - bbox._z_start]
	processing_crop_pads: [ [128 + 64, 128 + 64, 0]]
	//level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: #RELAXATION_RESOLUTION
	max_dist:       2

	tissue_mask: {
		"@type": "build_ts_layer"
		path:    #STAGE1_TISSUE_MASK_PATH
	}
	misalignment_masks: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#STAGE1_MISALIGNMENTS_PATH)/\(offset)"
				read_procs: [
					{
						"@type": "compare"
						"@mode": "partial"
						mode:    ">="
						value:   127
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			}
		}
	}
	pairwise_fields: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#FIELDS_PATH)/\(offset)"
			}
		}
	}
	pairwise_fields_inv: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#FIELDS_INV_PATH)/\(offset)"
			}
		}
	}
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)_z\(bbox._z_start)_\(bbox._z_end)"
	dst: {
		"@type": "build_volumetric_layer_set"
		layers: {
			match_offsets: {
				"@type":             "build_cv_layer"
				path:                match_offsets_path
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			img_mask: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/img_mask"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "erode", "@mode":     "partial", width: 4, thr:    3},
					{"@type": "filter_cc", "@mode": "partial", thr:   300, mode: "keep_large"},
					{"@type": "coarsen", "@mode":   "partial", width: 1, thr:    3},
					{"@type": "to_uint8", "@mode":  "partial"},
				]
			}
			aff_mask: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/aff_mask"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "erode", "@mode":     "partial", width: 4, thr:    3},
					{"@type": "filter_cc", "@mode": "partial", thr:   300, mode: "keep_large"},
					{"@type": "coarsen", "@mode":   "partial", width: 1, thr:    3},
					{"@type": "to_uint8", "@mode":  "partial"},
				]
			}
			sector_length_before: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/sl_before"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
			sector_length_after: {
				"@type":             "build_cv_layer"
				path:                "\(match_offsets_path)/sl_after"
				info_reference_path: #IMG_PATH
				info_chunk_size:     #RELAX_OUTCOME_CHUNK
				on_info_exists:      "overwrite"
				write_procs: [
					{"@type": "to_uint8", "@mode": "partial"},
				]
			}
		}
	}
}

#RELAX_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "AcedRelaxationOp"
	}
	expand_bbox_processing:    true
	dst_resolution: #RELAXATION_RESOLUTION
	bbox:           _

	processing_chunk_sizes: [[512, 512, bbox._z_end - bbox._z_start], [288, 288, bbox._z_end - bbox._z_start]]
	max_reduction_chunk_sizes: [512, 512, bbox._z_end - bbox._z_start]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	processing_blend_pads: [[32, 32, 0], [32, 32, 0]]
	//level_intermediaries_dirs: [#TMP_PATH, ]
	//              processing_chunk_sizes: [[32, 32, #Z_END - #Z_START], [28, 28, #Z_END - #Z_START]]
	//              max_reduction_chunk_sizes: [128, 128, #Z_END - #Z_START]
	//              processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	//              processing_blend_pads: [[12, 12, 0], [12, 12, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	fix:             #RELAXATION_FIX
	num_iter:        #RELAXATION_ITER
	lr:              #RELAXATION_LR
	rigidity_weight: #RELAXATION_RIG

	rigidity_masks: {
		"@type": "build_ts_layer"
		path:    #STAGE1_DEFECTS_PATH
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "=="
				value:   0
			},

		]
	}
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)_z\(bbox._z_start)_\(bbox._z_end)"
	match_offsets: {
		"@type": "build_ts_layer"
		path:    match_offsets_path
		//info_reference_path: #IMG_PATH
		on_info_exists: "overwrite"
	}
	pfields: {
		for offset in #Z_OFFSETS {
			"\(offset)": {
				"@type": "build_ts_layer"
				path:    "\(#FIELDS_PATH)/\(offset)"
			}
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #AFIELD_PATH
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			num_channels: 2
			data_type:    "float32"
		}
		info_chunk_size: #RELAX_OUTCOME_CHUNK
		on_info_exists:  "overwrite"
	}
}

#POST_ALIGN_FLOW: {
	_bbox: _
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)_z\(_bbox._z_start)_\(_bbox._z_end)"
	"@type": "mazepa.concurrent_flow"
	stages: [
		#WARP_FLOW_TMPL & {
			bbox: _bbox
			op: mode:    "img"
			src: path:   #IMG_PATH
			field: path: #AFIELD_PATH
			dst: path:   #IMG_ALIGNED_PATH
			dst_resolution: [32, 32, 45]
			field: data_resolution: #RELAXATION_RESOLUTION
		},

		#WARP_FLOW_TMPL & {
			bbox: _bbox
			op: mode:    "mask"
			src: path:   "\(match_offsets_path)/img_mask"
			field: path: #AFIELD_PATH
			dst: path:   #IMG_MASK_PATH
			dst_resolution: [64, 64, 45]
			field: data_resolution: #RELAXATION_RESOLUTION
		},
		#WARP_FLOW_TMPL & {
			bbox: _bbox
			op: mode:    "mask"
			src: path:   "\(match_offsets_path)/aff_mask"
			field: path: #AFIELD_PATH
			dst: path:   #AFF_MASK_PATH
			dst_resolution: [64, 64, 45]
			field: data_resolution: #RELAXATION_RESOLUTION
		},

	]
}

#RUN_INFERENCE: {
	"@type": "mazepa.execute_on_gcp_with_sqs"
	//worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x140"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x186"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	checkpoint:             "gs://zetta_utils_runs/sergiy/exec-fat-tapir-of-exotic-defiance/2023-05-05_040524_255226.zstd"
	worker_replicas:        10
	batch_gap_sleep_sec:    1
	do_dryrun_estimation:   true
	local_test:             false
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"

	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for block in #BLOCKS {
				let bbox = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
				"@type": "mazepa.sequential_flow"
				stages: [
					#MATCH_OFFSETS_FLOW & {'bbox': bbox},
					#RELAX_FLOW & {'bbox':         bbox},
					#POST_ALIGN_FLOW & {_bbox:     bbox},
				]
			},
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
