		// INPUTS
#IMG_PATH1: "gs://zetta-research-dodam/dacey-montaging-research/prototype_7104/(0,1)"
#IMG_PATH2: "gs://zetta-research-dodam/dacey-montaging-research/prototype_7104/(0,0)"

//#IMG_PATH1: "gs://zetta-research-dodam/dacey-montaging-research/prototype_7104_clahe/(0,1)"
//#IMG_PATH2: "gs://zetta-research-dodam/dacey-montaging-research/prototype_7104_clahe/(0,0)"
#ENC_PATH: #IMG_PATH1
#FOLDER:   "gs://zetta-research-dodam/dacey-montaging-research"
#IMG_RES: [5, 5, 50]

#IMG_SIZE: [65536, 65536, 16000]

//#SKIP_COMPUTE_FIELD: #SKIP_COMPUTE_FIELD & {
// "160": true
//}

//#SKIP_CF:           true
#SKIP_INVERT_FIELD: true

// #SKIP_WARP:         true
#SKIP_ENCODE: true
#SKIP_MISD:   true
#TEST_SMALL:  true
#TEST_LOCAL:  true

#CLUSTER_NUM_WORKERS: 128

// Use CNS model for these steps instead of zfish
//#STAGE_TMPL_160: src: path: #ENC_PATH_CNS
//#STAGE_TMPL_160: tgt: path: #ENC_PATH_CNS

// OUTPUTS
#COMBINED_FIELDS_PATH:  "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH:       "\(#FOLDER)/fields_inv"
#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped_500_200_100_50_20_4000iter_320-20"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

// PARAMETERS
#Z_OFFSETS: [0]

// #Z_OFFSETS: [-2]
#WARP_RES: [20, 20, #IMG_RES[2]]

// 375 - 525 test
#BBOX: {
	"@type": "BBox3D.from_coords"
	if #TEST_SMALL == false {
		start_coord: [0, 0, 0]
		end_coord:  #IMG_SIZE
		resolution: #IMG_RES
	}
	if #TEST_SMALL == true {
		// start_coord: [0, 0, 160*413]
		start_coord: [0, 0, 46 * 50]
		end_coord: [32768 * 5, 32768 * 5, start_coord[2] + 50]
	}
}

{
	"@type":               "mazepa.execute_on_gcp_with_sqs"
	worker_cluster_region: "us-east1"
	// worker_image:           "us.gcr.io/zetta-research/zetta_utils:tri-test-230829"
	// worker_cluster_project: "zetta-research"
	// worker_cluster_name:    "zutils-x3"
	worker_image:           "us.gcr.io/zetta-jkim-001/zetta_utils:tri-230830"
	worker_cluster_project: "zetta-jkim-001"
	worker_cluster_name:    "zutils"
	worker_resources: {
		memory:           "20000Mi" // sized for n1-highmem-4
		"nvidia.com/gpu": "1"
	}
	worker_replicas: #CLUSTER_NUM_WORKERS
	local_test:      #TEST_LOCAL
	target:          #JOINT_OFFSET_FLOW
}

#SKIP_CF:           _ | *false
#SKIP_INVERT_FIELD: _ | *false
#SKIP_WARP:         _ | *false
#SKIP_ENCODE:       _ | *false
#SKIP_MISD:         _ | *false
#TEST_SMALL:        _ | *false
#TEST_LOCAL:        _ | *false

// For testing multiple sections
// #TEST_SECTIONS: [1]
// #JOINT_OFFSET_FLOW: {
//     "@type": "mazepa.concurrent_flow"
//     stages: [
//         for z in #TEST_SECTIONS {
//             let bbox_ = {
//                 "@type": "BBox3D.from_coords"
//                 start_coord: [0, 0, z*160]
//                 end_coord: [262144, 262144, start_coord[2]+160]
//             }
//             #FLOW_ONE_SECTION & {_bbox: bbox_}
//         }
//     ]
// }
// For running one bbox
#JOINT_OFFSET_FLOW: #FLOW_ONE_SECTION & {_bbox: #BBOX}

#RESUME_CF_FLOW: _ | *false
#FLOW_ONE_SECTION: {
	_bbox:   _
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.sequential_flow"
			stages: [
				// Compute Field
				if #SKIP_CF == false {
					#CF_FLOW_TMPL & {
						bbox: _bbox
						dst: path: "\(#COMBINED_FIELDS_PATH)/\(z_offset)"
						tmp_layer_dir: "\(#COMBINED_FIELDS_PATH)/\(z_offset)/tmp"
						tgt_offset: [0, 0, z_offset]
					}
				},
				if #SKIP_INVERT_FIELD == false {
					// Invert Field
					#INVERT_FLOW_TMPL & {
						bbox: _bbox
						op_kwargs: src: path: "\(#COMBINED_FIELDS_PATH)/\(z_offset)"
						dst: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
					}
				},
				if #SKIP_WARP == false {
					// Warp Image with inverted field
					#WARP_FLOW_TMPL & {
						bbox: _bbox
						op: mode:  "img"
						dst: path: "\(#IMGS_WARPED_PATH)/\(z_offset)"
						op_kwargs: src: path: #IMG_PATH1
						op_kwargs: src: index_procs: [
							{
								"@type": "VolumetricIndexTranslator"
								offset: [0, 0, z_offset]
								resolution: #IMG_RES
							},
						]
						op_kwargs: field: path:            "\(#COMBINED_FIELDS_PATH)/\(z_offset)"
						op_kwargs: field: data_resolution: #STAGES[len(#STAGES)-1].dst_resolution
					}
				},
			]
		},
	]
}

#CF_FLOW_TMPL: {
	"@type":           "build_compute_field_multistage_flow"
	bbox:              _
	stages:            #STAGES
	src_offset?:       _
	tgt_offset?:       _
	src_field?:        _
	offset_resolution: #IMG_RES
	src: {
		"@type": "build_cv_layer"
		path:    _ | *#IMG_PATH1
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    _ | *#IMG_PATH2
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH1
		info_field_overrides: {
			"type":         "image"
			"data_type":    "float32"
			"num_channels": 2
		}
		on_info_exists: "overwrite"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: dst.path
		on_info_exists:      "overwrite"
	}
}

#STAGES: [
	#STAGE_TMPL_320 & {
		fn: sm:       1000
		fn: num_iter: 4000
	},
	#STAGE_TMPL_160 & {
		fn: sm:       500
		fn: num_iter: 4000
	},
	#STAGE_TMPL_80 & {
		fn: sm:       200
		fn: num_iter: 4000
	},
	#STAGE_TMPL_40 & {
		fn: sm:       100
		fn: num_iter: 4000
	},
	#STAGE_TMPL_20 & {
		fn: sm:       50
		fn: num_iter: 4000
	},
]
#STAGE_TMPL_320: #STAGE_TMPL & {
	dst_resolution: [320, 320, #IMG_RES[2]]
}
#STAGE_TMPL_160: #STAGE_TMPL & {
	dst_resolution: [160, 160, #IMG_RES[2]]
}
#STAGE_TMPL_80: #STAGE_TMPL & {
	dst_resolution: [80, 80, #IMG_RES[2]]
}
#STAGE_TMPL_40: #STAGE_TMPL & {
	dst_resolution: [40, 40, #IMG_RES[2]]
}
#STAGE_TMPL_20: #STAGE_TMPL & {
	dst_resolution: [40, 40, #IMG_RES[2]]
}

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	// processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	// skip_intermediaries: true
	// level_intermediaries_dirs: [#TMP_PATH, #TMP_PATH_LOCAL]
	expand_bbox_processing:  bool | *true
	shrink_processing_chunk: bool | *false
	src: {
		"@type":     "build_cv_layer"
		path:        _ | *#IMG_PATH1
		read_procs?: _
	}
	tgt: {
		"@type":     "build_cv_layer"
		path:        _ | *#IMG_PATH2
		read_procs?: _
	}
	fn: {
		"@type":        "align_with_online_finetuner"
		"@mode":        "partial"
		sm:             int | *10
		num_iter:       int | *200
		lr?:            float | *0.1
		bake_in_field?: bool
	}
}

#INVERT_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	// expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	skip_intermediaries: true
	dst_resolution:      #STAGES[len(#STAGES)-1].dst_resolution
	bbox:                _
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    _
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: op_kwargs.src.path
		info_chunk_size: [512, 512, 1]
		// info_add_scales_ref: "160_160_160"
		info_add_scales: [
			#STAGES[len(#STAGES)-1].dst_resolution,
		]
		info_add_scales_mode: "replace"
		on_info_exists:       "overwrite"
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	skip_intermediaries: true
	bbox:                _
	dst_resolution:      _ | *#WARP_RES
	op_kwargs: {
		src: {
			"@type":      "build_cv_layer"
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
		info_reference_path: _ | *#IMG_PATH1
		// info_add_scales_ref: "4_4_160"
		//  info_add_scales: [dst_resolution]
		//  info_add_scales_mode: "replace"
		info_chunk_size: [512, 512, 1]
		on_info_exists: "overwrite"
		write_procs?:   _
		index_procs?:   _ | *[]
	}
}
