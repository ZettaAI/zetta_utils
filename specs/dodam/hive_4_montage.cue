import "list"

import ( "math"

	// INPUTS
)

#ENCODED_PATH00: "\(#ENCODED_PATH_BASE)/0_0"
#ENCODED_PATH01: "\(#ENCODED_PATH_BASE)/0_1"
#ENCODED_PATH10: "\(#ENCODED_PATH_BASE)/1_0"
#ENCODED_PATH11: "\(#ENCODED_PATH_BASE)/1_1"

#IMG_PATH_BASE:     "gs://hive-tomography/pilot11-tiles/refactor-test-0"
#ENCODED_PATH_BASE: "gs://hive-tomography/pilot11-tiles/refactor-test-0-enc"

#IMG_PATH00: "\(#IMG_PATH_BASE)/0_0"
#IMG_PATH01: "\(#IMG_PATH_BASE)/0_1"
#IMG_PATH10: "\(#IMG_PATH_BASE)/1_0"
#IMG_PATH11: "\(#IMG_PATH_BASE)/1_1"

#IMG_RES: [4, 4, 1]

#IMG_SIZE: [786432, 262144, 1]

//#FOLDER2: "gs://hive-tomography/pilot11-montage/exp32-500-100c4-100"
#FOLDER: "gs://hive-tomography/pilot11-montage/refactor-test-0"

#SKIP_ENCODE: true
#SKIP_MISD:   true
#TEST_SMALL:  true
#TEST_LOCAL:  false

#CLUSTER_NUM_WORKERS: 100

#NUM_ITER: 500

#NUM_ITER_RELAX: 100
//#NUM_ITER: 0

// OUTPUTS

#FIELDS_PATH01: "\(#FOLDER)/fields_fwd_01"
#FIELDS_PATH11: "\(#FOLDER)/fields_fwd_11"
#FIELDS_PATH10: "\(#FOLDER)/fields_fwd_10"
#FIELDS_PATH00: "\(#FOLDER)/fields_fwd_00"

#FIELDS_RELAXED_PATH00: "\(#FOLDER)/fields_relaxed_00"
#FIELDS_RELAXED_PATH01: "\(#FOLDER)/fields_relaxed_01"
#FIELDS_RELAXED_PATH11: "\(#FOLDER)/fields_relaxed_11"
#FIELDS_RELAXED_PATH10: "\(#FOLDER)/fields_relaxed_10"

#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"
#IMG_WARPED_PATH01:     "\(#FOLDER)/img_warped/01"
#IMG_WARPED_PATH11:     "\(#FOLDER)/img_warped/11"
#IMG_WARPED_PATH10:     "\(#FOLDER)/img_warped/10"
#IMG_WARPED_PATH00:     "\(#FOLDER)/img_warped/00"
#IMG_WARPED_PATH_FINAL: "\(#FOLDER)/img_warped/final"

#ERRORS_PATH0001:   "\(#FOLDER)/errors/0001"
#ERRORS_PATH0111:   "\(#FOLDER)/errors/0111"
#ERRORS_PATH1110:   "\(#FOLDER)/errors/1110"
#ERRORS_PATH1000:   "\(#FOLDER)/errors/1000"
#ERRORS_PATH_FINAL: "\(#FOLDER)/errors/final"

// PARAMETERS
#Z_OFFSETS: [0]

#HIGHEST_RES: [16, 16, #IMG_RES[2]]
#LOWEST_RES: [512, 512, #IMG_RES[2]]
#IMG_WARP_RES: [16, 16, #IMG_RES[2]]
#ERROR_RES: [16, 16, #IMG_RES[2]]

//#NUM_WARP_DOWNSAMPLES: math.Log2(#LOWEST_RES[0] / #IMG_WARP_RES[0])
//#DOWNSAMPLE_WARP_RESES: [ for i in list.Range(0, NUM_WARP_DOWNSAMPLES) {(IMG_WARP_RES[0] * math.Pow(2, i+1))}]
#NUM_ERROR_DOWNSAMPLES: math.Log2(#ERROR_RES[0] / #IMG_WARP_RES[0])
#DOWNSAMPLE_ERROR_RESES: [ for i in list.Range(0, #NUM_ERROR_DOWNSAMPLES, 1) {(#IMG_WARP_RES[0] * math.Pow(2, i+1))}]
#NUM_ERROR_FINAL_DOWNSAMPLES: math.Log2(#LOWEST_RES[0] / #ERROR_RES[0])
#DOWNSAMPLE_ERROR_FINAL_RESES: [ for i in list.Range(0, #NUM_ERROR_FINAL_DOWNSAMPLES, 1) {(#ERROR_RES[0] * math.Pow(2, i+1))}]
#NUM_IMG_DOWNSAMPLES: math.Log2(#LOWEST_RES[0] / #IMG_WARP_RES[0])
#DOWNSAMPLE_IMG_RESES: [ for i in list.Range(0, #NUM_IMG_DOWNSAMPLES, 1) {(#IMG_WARP_RES[0] * math.Pow(2, i+1))}]

// 375 - 525 test
#BBOX: {
	"@type": "BBox3D.from_coords"
	if #TEST_SMALL == false {
		start_coord: [0, 0, 0]
		end_coord:  #IMG_SIZE
		resolution: #IMG_RES
	}
	if #TEST_SMALL == true {
		start_coord: [0, 0, 0]
		end_coord: [32768, 16384, 2]
		resolution: [16, 16, 1]
	}
}

{
	"@type":               "mazepa.execute_on_gcp_with_sqs"
	worker_cluster_region: "us-east1"
	worker_image:          "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-test-montaging-refactor-12"
	worker_resources: {
		memory:           "18560Mi" // sized for n1-highmem-4
		"nvidia.com/gpu": "1"
	}
	worker_cluster_project: "zetta-research"
	worker_cluster_name:    "zutils-x3"
	worker_replicas:        #CLUSTER_NUM_WORKERS
	local_test:             #TEST_LOCAL
	debug:                  #TEST_LOCAL
	target:                 #JOINT_OFFSET_FLOW
	num_procs:              1
	semaphores_spec: {
		"read":  1
		"cpu":   1
		"cuda":  1
		"write": 1
	}
}

#SKIP_CF:           _ | *false
#SKIP_INVERT_FIELD: _ | *false

#SKIP_WARP:   _ | *false //
#SKIP_ENCODE: _ | *false
#SKIP_MISD:   _ | *false
#TEST_SMALL:  _ | *false
#TEST_LOCAL:  _ | *false

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
				//    {"@type": "mazepa.concurrent_flow"
				//     stages: [
				//      #COMBINED_FLOW_TMPL & {
				//       _bbox_:       _bbox
				//       _encoded_src: #ENCODED_PATH00
				//       _encoded_tgt: #ENCODED_PATH01
				//       _src_field:   #FIELDS_PATH00
				//      },
				//      #COMBINED_FLOW_TMPL & {
				//       _bbox_:       _bbox
				//       _encoded_src: #ENCODED_PATH01
				//       _encoded_tgt: #ENCODED_PATH11
				//       _src_field:   #FIELDS_PATH01
				//      },
				//      #COMBINED_FLOW_TMPL & {
				//       _bbox_:       _bbox
				//       _encoded_src: #ENCODED_PATH11
				//       _encoded_tgt: #ENCODED_PATH10
				//       _src_field:   #FIELDS_PATH11
				//      },
				//      #COMBINED_FLOW_TMPL & {
				//       _bbox_:       _bbox
				//       _encoded_src: #ENCODED_PATH10
				//       _encoded_tgt: #ENCODED_PATH00
				//       _src_field:   #FIELDS_PATH10
				//      },
				//     ]
				//    },
				//           #MR_FLOW_TMPL & {
				//            bbox:           _bbox
				//            dst_resolution: #HIGHEST_RES
				//           },
				{"@type": "mazepa.concurrent_flow"
					stages: [
						#WARP_FLOW_TMPL & {
							bbox: _bbox
							op: mode:                 "img"
							dst: path:                #IMG_WARPED_PATH00
							dst: info_reference_path: #IMG_PATH00
							dst_resolution: #IMG_WARP_RES
							op_kwargs: src: path:              #IMG_PATH00
							op_kwargs: field: path:            #FIELDS_RELAXED_PATH00
							op_kwargs: field: data_resolution: #HIGHEST_RES
						},
						#WARP_FLOW_TMPL & {
							bbox: _bbox
							op: mode:                 "img"
							dst: path:                #IMG_WARPED_PATH01
							dst: info_reference_path: #IMG_PATH01
							dst_resolution: #IMG_WARP_RES
							op_kwargs: src: path:              #IMG_PATH01
							op_kwargs: field: path:            #FIELDS_RELAXED_PATH01
							op_kwargs: field: data_resolution: #HIGHEST_RES
						},
						#WARP_FLOW_TMPL & {
							bbox: _bbox
							op: mode:                 "img"
							dst: path:                #IMG_WARPED_PATH11
							dst: info_reference_path: #IMG_PATH11
							dst_resolution: #IMG_WARP_RES
							op_kwargs: src: path:              #IMG_PATH11
							op_kwargs: field: path:            #FIELDS_RELAXED_PATH11
							op_kwargs: field: data_resolution: #HIGHEST_RES
						},
						#WARP_FLOW_TMPL & {
							bbox: _bbox
							op: mode:                 "img"
							dst: path:                #IMG_WARPED_PATH10
							dst: info_reference_path: #IMG_PATH10
							dst_resolution: #IMG_WARP_RES
							op_kwargs: src: path:              #IMG_PATH10
							op_kwargs: field: path:            #FIELDS_RELAXED_PATH10
							op_kwargs: field: data_resolution: #HIGHEST_RES
						},
					]
				},
				// Combine Warped Images
				#COMBINE_IMAGE_FLOW_TMPL & {
					bbox: _bbox
					dst: layers: output: path:                #IMG_WARPED_PATH_FINAL
					dst: layers: output: info_reference_path: #IMG_PATH00
					dst: layers: errors: path:                #ERRORS_PATH_FINAL
					dst: layers: errors: info_reference_path: #IMG_PATH00
					dst_resolution: #IMG_WARP_RES
					op_kwargs: data1: path: #IMG_WARPED_PATH00
					op_kwargs: data2: path: #IMG_WARPED_PATH01
					op_kwargs: data3: path: #IMG_WARPED_PATH11
					op_kwargs: data4: path: #IMG_WARPED_PATH10
				},
				{"@type": "mazepa.concurrent_flow"
					stages: [
						{"@type": "mazepa.sequential_flow"
							// Downsample Combined Images
							stages: [
								for res in #DOWNSAMPLE_IMG_RESES {
									#DOWNSAMPLE_FLOW_TMPL & {
										bbox: _bbox
										op: mode: "img"
										op_kwargs: src: path:                #IMG_WARPED_PATH_FINAL
										op_kwargs: src: info_reference_path: #IMG_PATH00
										dst: path:                #IMG_WARPED_PATH_FINAL
										dst: info_reference_path: #IMG_PATH00
										dst_resolution: [res, res, #IMG_RES[2]]
									}
								},
							]
						},

						{"@type": "mazepa.sequential_flow"
							stages: [
								// Downsample Error Map
								for res in #DOWNSAMPLE_IMG_RESES {
									#DOWNSAMPLE_FLOW_TMPL & {
										bbox: _bbox
										op: mode: "img"
										op_kwargs: src: path:                #ERRORS_PATH_FINAL
										op_kwargs: src: info_reference_path: #IMG_PATH00
										dst: path:                #ERRORS_PATH_FINAL
										dst: info_reference_path: #IMG_PATH00
										dst_resolution: [res, res, #IMG_RES[2]]
									}
								},

							]
						},
					]
				},
			]
		},
	]
}

#COMBINED_FLOW_TMPL: {
	"@type":      "mazepa.sequential_flow"
	_bbox_:       _
	_encoded_src: _
	_encoded_tgt: _
	_src_field:   _
	stages: [
		// Compute Field
		#CF_FLOW_TMPL & {
			bbox: _bbox_
			src: path: _encoded_src
			tgt: path: _encoded_tgt
			dst: path: _src_field
			tmp_layer_dir: "\(_src_field)/tmp"
			tgt_offset: [0, 0, 0]
		},
	]
}
#MR_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "MontagingRelaxOperation"
	}
	processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1], [1024 * 1, 1024 * 1, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	// processing_crop_pads: [[0, 0, 0], [512, 512, 0]]
	// processing_blend_pads: [[512, 512, 0], [128, 128, 0]]
	//                   processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	skip_intermediaries: true
	bbox:                _
	dst_resolution:      _
	op_kwargs: {
		fields: [
			{
				"@type":            "build_cv_layer"
				path:               #FIELDS_PATH00
				data_resolution:    _ | *null
				interpolation_mode: "field"
			},
			{
				"@type":            "build_cv_layer"
				path:               #FIELDS_PATH01
				data_resolution:    _ | *null
				interpolation_mode: "field"
			},
			{
				"@type":            "build_cv_layer"
				path:               #FIELDS_PATH11
				data_resolution:    _ | *null
				interpolation_mode: "field"
			},
			{
				"@type":            "build_cv_layer"
				path:               #FIELDS_PATH10
				data_resolution:    _ | *null
				interpolation_mode: "field"
			},
		]
		srcs: [
			{
				"@type":         "build_cv_layer"
				path:            #IMG_PATH00
				data_resolution: _ | *null
			},
			{
				"@type":         "build_cv_layer"
				path:            #IMG_PATH01
				data_resolution: _ | *null
			},
			{
				"@type":         "build_cv_layer"
				path:            #IMG_PATH11
				data_resolution: _ | *null
			},
			{
				"@type":         "build_cv_layer"
				path:            #IMG_PATH10
				data_resolution: _ | *null
			},
		]
		num_iter:        #NUM_ITER_RELAX
		rigidity_weight: 100.0
		rigidity_scales: [1, 2, 4]
		scales_to_relax: [1]
		src_mask_gaussian_blur_sigma: 5
		// for generating lens correction: rigidity_weight: 1000.0
	}
	dst: {
		"@type": "build_volumetric_layer_set"
		layers:
		{
			"0": {
				"@type":             "build_cv_layer"
				path:                #FIELDS_RELAXED_PATH00
				info_reference_path: #FIELDS_PATH00
				info_chunk_size: [512, 512, 1]
				on_info_exists: "overwrite"
				write_procs?:   _
				index_procs?:   _ | *[]
			}
			"1": {
				"@type":             "build_cv_layer"
				path:                #FIELDS_RELAXED_PATH01
				info_reference_path: #FIELDS_PATH01
				info_chunk_size: [512, 512, 1]
				on_info_exists: "overwrite"
				write_procs?:   _
				index_procs?:   _ | *[]
			}
			"2": {
				"@type":             "build_cv_layer"
				path:                #FIELDS_RELAXED_PATH11
				info_reference_path: #FIELDS_PATH11
				info_chunk_size: [512, 512, 1]
				on_info_exists: "overwrite"
				write_procs?:   _
				index_procs?:   _ | *[]
			}
			"3": {
				"@type":             "build_cv_layer"
				path:                #FIELDS_RELAXED_PATH10
				info_reference_path: #FIELDS_PATH10
				info_chunk_size: [512, 512, 1]
				on_info_exists: "overwrite"
				write_procs?:   _
				index_procs?:   _ | *[]
			}
		}

	}
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
		path:    _
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #IMG_PATH00
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
	#CF_FINETUNER_STAGE_TMPL & {
		fn: sm:       500
		fn: num_iter: #NUM_ITER
		dst_resolution: [512, 512, #IMG_RES[2]]
		processing_chunk_sizes: [[1024, 1024, 1], [512, 512, 1]]
	},
	#CF_FINETUNER_STAGE_TMPL & {
		fn: sm:       250
		fn: num_iter: #NUM_ITER
		dst_resolution: [256, 256, #IMG_RES[2]]
		processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1], [1024, 1024, 1]]
	},
	#CF_FINETUNER_STAGE_TMPL & {
		fn: sm:       100
		fn: num_iter: #NUM_ITER
		dst_resolution: [128, 128, #IMG_RES[2]]
		processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1], [1024, 1024, 1]]
	},
	#CF_FINETUNER_STAGE_TMPL & {
		fn: sm:       25
		fn: num_iter: #NUM_ITER
		dst_resolution: [64, 64, #IMG_RES[2]]
		processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1], [1024, 1024, 1]]
	},
	#CF_FINETUNER_STAGE_TMPL & {
		fn: sm:       10
		fn: num_iter: #NUM_ITER
		dst_resolution: [32, 32, #IMG_RES[2]]
		processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	},
	#CF_FINETUNER_STAGE_TMPL & {
		fn: sm:       5
		fn: num_iter: #NUM_ITER
		dst_resolution: [16, 16, #IMG_RES[2]]
		processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	},
]

#CF_FINETUNER_STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	// processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_chunk_sizes: _
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	// skip_intermediaries:     true
	//                       level_intermediaries_dirs: [#TMP_PATH, #TMP_PATH_LOCAL]
	expand_bbox_processing:  bool | *true
	shrink_processing_chunk: bool | *false
	fn: {
		"@type":        "align_with_online_finetuner"
		"@mode":        "partial"
		sm:             _
		num_iter:       _
		lr?:            float | *0.1
		bake_in_field?: bool
	}
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [128, 128, 0]]
	skip_intermediaries: true
	bbox:                _
	dst_resolution:      _
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
		info_reference_path: _
		info_chunk_size: [512, 512, 1]
		on_info_exists: "overwrite"
		write_procs?:   _
		index_procs?:   _ | *[]
	}
}
#COMBINE_IMAGE_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "ComposeWithErrorsOperation"
	}
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	skip_intermediaries: true
	bbox:                _
	dst_resolution:      _
	op_kwargs: {
		data1: {
			"@type":      "build_cv_layer"
			path:         _
			read_procs?:  _
			index_procs?: _ | *[]
		}
		data2: {
			"@type":      "build_cv_layer"
			path:         _
			read_procs?:  _
			index_procs?: _ | *[]
		}
		data3: {
			"@type":      "build_cv_layer"
			path:         _
			read_procs?:  _
			index_procs?: _ | *[]
		}
		data4: {
			"@type":      "build_cv_layer"
			path:         _
			read_procs?:  _
			index_procs?: _ | *[]
		}
		erosion: 15
	}
	dst: {
		"@type": "build_volumetric_layer_set"
		layers:
		{
			output: {
				"@type":             "build_cv_layer"
				path:                _
				info_reference_path: _
				// info_add_scales_ref: "4_4_160"
				//  info_add_scales: [dst_resolution]
				//  info_add_scales_mode: "replace"
				info_chunk_size: [512, 512, 1]
				on_info_exists: "overwrite"
				write_procs?:   _
				index_procs?:   _ | *[]
			}
			errors: {
				"@type":             "build_cv_layer"
				path:                _
				info_reference_path: _
				// info_add_scales_ref: "4_4_160"
				//  info_add_scales: [dst_resolution]
				//  info_add_scales_mode: "replace"
				info_chunk_size: [512, 512, 1]
				on_info_exists: "overwrite"
				write_procs?:   _
				index_procs?:   _ | *[]
			}
		}
	}
}

#DOWNSAMPLE_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	//expand_bbox_processing: true
	shrink_processing_chunk: false
	skip_intermediaries:     true
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	op: {
		"@type":         "InterpolateOperation"
		mode:            _ | "img"
		res_change_mult: _ | *[2, 2, 1]
	}
	bbox: _
	op_kwargs: {
		src: {
			"@type":             "build_cv_layer"
			path:                _
			read_procs:          _ | *[]
			info_reference_path: _
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: _
	}
	dst_resolution: _
} //
