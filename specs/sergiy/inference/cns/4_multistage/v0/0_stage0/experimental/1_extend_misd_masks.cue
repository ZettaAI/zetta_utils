#IMG_PATH:     "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1/raw_img"
#BASE_FOLDER:  "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0"
#MED_IMG_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0/img_aligned_1024nm_try_x6_iter8000_rig0.5_lr0.005"
#TMP_PATH:     "gs://tmp_2s/yo/"

//OUTPUTS
#PAIRWISE_SUFFIX: "try_x0"

//#FIELDS_PATH:       "\(#BASE_FOLDER)/fields_fwd"
//#RIGIDITY_MAP_PATH: "\(#BASE_FOLDER)/rigidity_fwd"

#FIELDS_BASE: "\(#BASE_FOLDER)/fields_"

#STRETCH_MASK_BASE: "\(#BASE_FOLDER)/stretch_"

#MISALIGNMENTS_PATH:          "\(#BASE_FOLDER)/misalignments"
#MISALIGNMENTS_COMBINED_PATH: "\(#BASE_FOLDER)/misalignments_combined"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2900]
	end_coord: [2048, 2048, 3000]
	resolution: [512, 512, 45]
}

#GET_RIGIDITY_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "get_rigidity_map", "@mode": "partial"}
	expand_bbox_processing: true

	// processing_chunk_sizes: [[4 * 1024, 4 * 1024, 1], [1024, 1024, 1]]
	// max_reduction_chunk_sizes: [4 * 1024, 4 * 1024, 1]
	// processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	// level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	processing_chunk_sizes: [[6 * 1024, 6 * 1024, 1]]
	processing_crop_pads: [[128, 128, 0]]

	dst_resolution: [32, 32, 45]
	bbox: #BBOX
	field: {
		"@type": "build_ts_layer"
		path:    "\(#FIELDS_BASE)\(_field_type)/\(_z_offset)"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#STRETCH_MASK_BASE)\(_field_type)/\(_z_offset)"
		info_reference_path: field.path
		info_field_overrides: {
			num_channels: 1
			data_type:    "uint8"
		}
		write_procs: [
			{
				"@type":     "split_reduce"
				"@mode":     "partial"
				reduce_mode: "maximum"
				paths: [
					{
						"@type":        "chain"
						"@mode":        "partial"
						chain_arg_name: "data"
						callables: [
							{"@type": "compare", "@mode":   "partial", mode:  ">=", value: 0.7},
							{"@type": "coarsen", "@mode":   "partial", width: 2, thr:      3},
							{"@type": "erode", "@mode":     "partial", width: 2, thr:      3},
							{"@type": "filter_cc", "@mode": "partial", thr:   300, mode:   "keep_large"},
							{"@type": "coarsen", "@mode":   "partial", width: 2, thr:      3},
						]
					},
					{
						"@type":        "chain"
						"@mode":        "partial"
						chain_arg_name: "data"
						callables: [
							{"@type": "compare", "@mode":   "partial", mode:  ">=", value: 0.35},
							{"@type": "coarsen", "@mode":   "partial", width: 5, thr:      3},
							{"@type": "erode", "@mode":     "partial", width: 5, thr:      3},
							{"@type": "filter_cc", "@mode": "partial", thr:   1500, mode:  "keep_large"},
							{"@type": "coarsen", "@mode":   "partial", width: 1, thr:      3},
						]
					},
					{
						"@type":        "chain"
						"@mode":        "partial"
						chain_arg_name: "data"
						callables: [
							{"@type": "compare", "@mode":   "partial", mode:  ">=", value: 0.2},
							{"@type": "coarsen", "@mode":   "partial", width: 4, thr:      3},
							{"@type": "erode", "@mode":     "partial", width: 4, thr:      3},
							{"@type": "filter_cc", "@mode": "partial", thr:   4000, mode:  "keep_large"},
						]
					},
				]
			},
			{"@type": "multiply", "@mode": "partial", value: 255},
			{"@type": "to_uint8", "@mode": "partial"},
		]
	}
	_z_offset:   int
	_field_type: string
}

#WARP_MASK_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "mask"
	}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	bbox: #BBOX
	dst_resolution: [32, 32, 45]
	src: {
		"@type": "build_ts_layer"
		path:    "\(#STRETCH_MASK_BASE)fwd/\(_z_offset)"
	}
	field: {
		"@type": "build_ts_layer"
		path:    "\(#FIELDS_BASE)inv/\(_z_offset)"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#STRETCH_MASK_BASE)fwd_warped/\(_z_offset)"
		info_reference_path: src.path
		on_info_exists:      "overwrite"
	}
	_z_offset: int
}

#COMBINE_MASK_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "apply_mask_fn"
		"@mode":    "partial"
		fill_value: 255.0
	}
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	//chunk_size: [512, 512, 1]
	bbox: #BBOX
	dst_resolution: [32, 32, 45]
	src: {
		"@type": "build_ts_layer"
		path:    "\(#MISALIGNMENTS_PATH)/\(_z_offset)"
	}
	masks: [
		{
			"@type": "build_ts_layer"
			path:    "\(#STRETCH_MASK_BASE)fwd_warped/\(_z_offset)"
		},
		{
			"@type": "build_ts_layer"
			path:    "\(#STRETCH_MASK_BASE)inv/\(_z_offset)"
		},

	]
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#MISALIGNMENTS_COMBINED_PATH)/\(_z_offset)"
		info_reference_path: src.path
	}
	_z_offset: int
}
"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x164"
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

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in [-1, -2] {
			[
				{
					"@type": "mazepa.sequential_flow"
					stages: [
						{
							"@type": "mazepa.concurrent_flow"
							stages: [
								#GET_RIGIDITY_FLOW & {_z_offset: z_offset, _field_type: "inv"},
								{
									"@type": "mazepa.sequential_flow"
									stages: [
										#GET_RIGIDITY_FLOW & {_z_offset: z_offset, _field_type: "fwd"},
										#WARP_MASK_TMPL & {_z_offset:    z_offset},
									]
								},
							]
						},
						#COMBINE_MASK_TMPL & {_z_offset: z_offset},
					]
				},
			]
		},
	]
}
