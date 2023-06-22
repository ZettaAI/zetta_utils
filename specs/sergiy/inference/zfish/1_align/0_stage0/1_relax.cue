// R 2UN ACED BLOCK
// INPUTS
#BASE_FOLDER: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/aced"

#IMG_PATH:         "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip3_img_defects_masked"
#DEFECTS_PATH:     "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"
#TISSUE_MASK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/tissue_mask"

#TMP_PATH: "gs://tmp_2w/temporary_layers"

//OUTPUTS
#PAIRWISE_SUFFIX: "final_x0"

#FOLDER:          "\(#BASE_FOLDER)/med_x1/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments2"

#AFIELD_PATH:      "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH: "\(#FOLDER)/img_aligned\(#RELAXATION_SUFFIX)"
#IMG_MASK_PATH:    "\(#FOLDER)/img_mask\(#RELAXATION_SUFFIX)"
#AFF_MASK_PATH:    "\(#FOLDER)/aff_mask\(#RELAXATION_SUFFIX)"

#IMG_ALIGNED_MASKED_PATH: "\(#FOLDER)/img_aligned_masked\(#RELAXATION_SUFFIX)"

//#MATCH_OFFSETS_BASE: "\(#FOLDER)/match_offsets_z\(#Z_START)_\(#Z_END)"
#MATCH_OFFSET_BASE: "\(#FOLDER)/match_offsets_\(#RELAXATION_RESOLUTION[0])nm_v4_z"

//#BASE_INFO_CHUNK: [128, 128, 1]
#BASE_INFO_CHUNK: [512, 512, 1]
#RELAX_OUTCOME_CHUNK: [32, 32, 1]
#RELAXATION_ITER:      8000
#RELAXATION_LR:        1e-3
#RELAXATION_GRAD_CLIP: 0.01

#RELAXATION_RIG: 0.5

//#Z_END:   746
#DEBUG_SUFFIX: ""

//#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"
#RELAXATION_RESOLUTION: [512, 512, 30]
#RELAXATION_SUFFIX: "try_x8_\(#RELAXATION_RESOLUTION[0])nm_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_lr\(#RELAXATION_LR)_clip\(#RELAXATION_GRAD_CLIP)\(#DEBUG_SUFFIX)"
#BLOCKS: [
	//{_z_start: 2958, _z_end: 3092, _fix: "first"}, // cutout G
	//{_z_start: 2958, _z_end: 2965, _fix: "first"}, // debug masks
	//{_z_start: 3910, _z_end: 3920, _fix: "first"}, // debug masks

	{_z_start: 3597, _z_end: 4014, _fix: "first"},
	{_z_start: 3597, _z_end: 4014, _fix: "first"},
	{_z_start: 2701, _z_end: 3598, _fix: "both"},
	{_z_start: 1805, _z_end: 2702, _fix: "both"},
	{_z_start: 892, _z_end:  1806, _fix: "both"},
	{_z_start: 0, _z_end:    893, _fix:  "last"},
]

#BBOX_TMPL: {
	"@type":  "BBox3D.from_coords"
	_z_start: int
	_z_end:   int
	start_coord: [0, 0, _z_start]
	//end_coord: [192 + 32, 256 + 32, _z_end]
	//start_coord: [0, 0, _z_start]
	end_coord: [384, 512, _z_end]
	resolution: [1024, 1024, 30]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [52189, 67314, 1025]
	scale_bar_nm: 30000
	layers: [
		["precoarse_img", "image", "precomputed://\(#IMG_PATH)"],
		//["-1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/-1"],
		//["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["afield \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#AFIELD_PATH)"],
	]
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	expand_bbox: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	//chunk_size: [512, 512, 1]
	bbox:           _
	dst_resolution: _ | *[32, 32, 30]
	op_kwargs: {
		src: {
			"@type":      "build_ts_layer"
			path:         _
			read_procs?:  _
			index_procs?: _ | *[]
		}
		field: {
			"@type":            "build_cv_layer"
			path:               _
			data_resolution:    _ | *null
			interpolation_mode: "field"
		}
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

#Z_OFFSETS: [-2, -1]

#MATCH_OFFSETS_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "AcedMatchOffsetOp"
	}
	bbox: _

	processing_chunk_sizes: [[32, 32, bbox._z_end - bbox._z_start]]
	processing_crop_pads: [[32, 32, 0]]
	dst_resolution: #RELAXATION_RESOLUTION
	op_kwargs: {
		max_dist: 2
		tissue_mask: {
			"@type": "build_ts_layer"
			path:    #TISSUE_MASK_PATH
		}
		misalignment_masks: {
			for offset in #Z_OFFSETS {
				"\(offset)": {
					"@type": "build_ts_layer"
					path:    "\(#MISALIGNMENTS_PATH)/\(offset)"
					read_procs: [
						{
							"@type": "compare"
							"@mode": "partial"
							mode:    ">="
							value:   128
						},
						{
							"@type": "filter_cc"
							"@mode": "partial"
							thr:     10
							mode:    "keep_large"
						},
						{
							"@type": "binary_closing"
							"@mode": "partial"
							width:   4
						},
						{
							"@type": "coarsen"
							"@mode": "partial"
							width:   1
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
					"@type": "build_cv_layer"
					path:    "\(#FIELDS_PATH)/\(offset)"
				}
			}
		}
		pairwise_fields_inv: {
			for offset in #Z_OFFSETS {
				"\(offset)": {
					"@type": "build_cv_layer"
					path:    "\(#FIELDS_INV_PATH)/\(offset)"
				}
			}
		}

	}
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)\(bbox._z_start)_\(bbox._z_end)"
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
					{"@type": "filter_cc", "@mode": "partial", thr: 3, mode: "keep_large"},
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
					{"@type": "filter_cc", "@mode": "partial", thr: 3, mode: "keep_large"},
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
	expand_bbox:    true
	dst_resolution: #RELAXATION_RESOLUTION
	bbox:           _

	processing_chunk_sizes: [[32, 32, bbox._z_end - bbox._z_start]]
	max_reduction_chunk_sizes: [32, 32, bbox._z_end - bbox._z_start]
	processing_crop_pads: [[24, 24, 0]]
	processing_blend_pads: [[16, 16, 0]]
	level_intermediaries_dirs: [#TMP_PATH]
	//              processing_chunk_sizes: [[32, 32, #Z_END - #Z_START], [28, 28, #Z_END - #Z_START]]
	//              max_reduction_chunk_sizes: [128, 128, #Z_END - #Z_START]
	//              processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	//              processing_blend_pads: [[12, 12, 0], [12, 12, 0]]
	//level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	op_kwargs: {
		fix:                     _
		num_iter:                #RELAXATION_ITER
		lr:                      #RELAXATION_LR
		grad_clip:               #RELAXATION_GRAD_CLIP
		rigidity_weight:         #RELAXATION_RIG
		min_rigidity_multiplier: 0.001

		rigidity_masks: {
			"@type": "build_ts_layer"
			path:    #DEFECTS_PATH
			read_procs: [
				{
					"@type": "compare"
					"@mode": "partial"
					mode:    "!="
					value:   0
				},
				{
					"@type": "filter_cc"
					"@mode": "partial"
					mode:    "keep_large"
					thr:     20
				},
				{
					"@type": "compare"
					"@mode": "partial"
					mode:    "=="
					value:   0
				},
			]
		}
		let match_offsets_path = "\(#MATCH_OFFSET_BASE)\(bbox._z_start)_\(bbox._z_end)"
		match_offsets: {
			"@type": "build_ts_layer"
			path:    match_offsets_path

			//info_reference_path: #IMG_PATH
			on_info_exists: "overwrite"
		}
		pfields: {
			for offset in #Z_OFFSETS {
				"\(offset)": {
					"@type": "build_cv_layer"
					path:    "\(#FIELDS_PATH)/\(offset)"
				}
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

#DOWNSAMPLE_FLOW_TMPL: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	dst_resolution: _
	op: {
		"@type": "InterpolateOperation"
		mode:    _
		res_change_mult: [2, 2, 1]
	}
	bbox: _
	op_kwargs: {
		src: {
			"@type":    "build_ts_layer"
			path:       _
			read_procs: _ | *[]
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    op_kwargs.src.path
	}
}

#DOWNSAMPLE_FLOW: {
	_bbox:   _
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode: "img" // not thresholded due to subhcunkable bug
								op_kwargs: src: path: "\(#MISALIGNMENTS_PATH)/\(z_offset)"
								// src: read_procs: [
								//  {"@type": "filter_cc", "@mode": "partial", mode: "keep_large", thr: 20},
								// ]
								dst_resolution: [res, res, 30]
							}
						},
					]
				},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode: "mask"
								op_kwargs: src: path: #TISSUE_MASK_PATH
								dst_resolution: [res, res, 30]
							}
						},
					]
				},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode: "field"
								op_kwargs: src: path: "\(#FIELDS_PATH)/\(z_offset)"
								dst_resolution: [res, res, 30]
							}
						},
					]
				},
				{
					"@type": "mazepa.seq_flow"
					stages: [
						for res in [64, 128, 256, 512, 1024] {
							#DOWNSAMPLE_FLOW_TMPL & {
								bbox: _bbox
								op: mode: "field"
								op_kwargs: src: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
								dst_resolution: [res, res, 30]
							}
						},
					]
				},
			]
		},
	]
}

#POST_ALIGN_FLOW: {
	_bbox:   _
	"@type": "mazepa.concurrent_flow"
	let match_offsets_path = "\(#MATCH_OFFSET_BASE)\(_bbox._z_start)_\(_bbox._z_end)"
	stages: [
		// #WARP_FLOW_TMPL & {
		//  bbox: _bbox
		//  op: mode: "img"
		//  op_kwargs: src: path:              #IMG_PATH
		//  op_kwargs: field: path:            #AFIELD_PATH
		//  op_kwargs: field: data_resolution: #RELAXATION_RESOLUTION
		//  dst: path: #IMG_ALIGNED_PATH
		//  dst_resolution: [32, 32, 30]
		// }
		#WARP_FLOW_TMPL & {
			op: mode: "mask"
			op_kwargs: {
				src: path:              "\(match_offsets_path)/img_mask"
				field: data_resolution: #RELAXATION_RESOLUTION
				field: path:            #AFIELD_PATH
			}
			dst: path: #IMG_MASK_PATH
			dst_resolution: [512, 512, 30]
			bbox: _bbox
		},
		#WARP_FLOW_TMPL & {
			op: mode: "mask"
			bbox: _bbox
			op_kwargs: {
				src: path:              "\(match_offsets_path)/aff_mask"
				field: data_resolution: #RELAXATION_RESOLUTION
				field: path:            #AFIELD_PATH
			}
			dst: path: #AFF_MASK_PATH
			dst_resolution: [512, 512, 30]
		},

	]
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x207"

	worker_resources: {
		memory: "18560Mi"
		//"nvidia.com/gpu": "1"
	}
	worker_replicas:        400
	do_dryrun_estimation:   true
	local_test:             false
	worker_cluster_name:    "zutils-zfish"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-jlichtman-zebrafish-001"
	checkpoint:             "gs://zetta_utils_runs/sergiy/exec-beryl-chachalaca-of-immortal-courage/2023-05-25_044359_601.zstd"
	target: {
		//"@type": "mazepa.concurrent_flow"
		"@type": "mazepa.seq_flow"
		stages: [
			for block in #BLOCKS {
				let bbox = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
				"@type": "mazepa.seq_flow"
				stages: [
					// #JOINT_OFFSET_FLOW & {
					//  _bbox: bbox
					// },
					// #CREATE_TISSUE_MASK & {
					//  'bbox': bbox
					// },
					// #DOWNSAMPLE_FLOW & {
					//  _bbox: bbox
					// }
					//#MATCH_OFFSETS_FLOW & {'bbox': bbox}
					//#RELAX_FLOW & {'bbox':         bbox, op_kwargs: fix: block._fix},
					#POST_ALIGN_FLOW & {_bbox: bbox},
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
