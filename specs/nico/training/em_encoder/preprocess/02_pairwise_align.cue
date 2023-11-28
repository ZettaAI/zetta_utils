import "math"
import "list"

#BASE_FOLDER: "zetta-research-nico/encoder/"
#IMG_SRC_PATH: "\(#BASE_FOLDER)/datasets/"
#IMG_DST_PATH: "\(#BASE_FOLDER)/pairwise_aligned/"

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1]

#FIELD_INFO_OVERRIDE: {
	"data_type": "float32",
	"num_channels": 2,
	"scales": [
		for i in list.Range(0, 10, 1) {
			encoding:     "zfpc"
			zfpc_correlated_dims: [true, true, false, false]
			zfpc_tolerance: 0.001953125
		}
	],
	"type": "image"
}

#ALIGN_STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [128, 128, 45]

		fn: {
			sm:       10
			num_iter: 500
			lr:       0.05
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [64, 64, 45]

		fn: {
			sm:       10
			num_iter: 300
			lr:       0.1
		}
		chunk_size: [2048, 2048, 1]
	},

	#STAGE_TMPL & {
		dst_resolution: [32, 32, 45]

		fn: {
			sm:       10
			num_iter: 200
			lr:       0.1
		}
		chunk_size: [2048, 2048, 1]
	},
]

#ENCODE_STAGES: [
	{
		type: "BaseEncoder"
		model: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"
		resolution: [32, 32, 45]
		res_change_mult: [1, 1, 1]
	},
]

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	chunk_size:     _
	fn: {
		"@type":  "align_with_online_finetuner"
		"@mode":  "partial"
		sm:       _
		num_iter: _
		lr?:      _
	}
	crop_pad: [64, 64, 0]
}


#CF_FLOW_TMPL: {
	"@type":     "build_compute_field_multistage_flow"
	bbox:        #BCUBE_COMBINED_32NM
	stages:      _
	src_offset?: _
	tgt_offset?: _
	offset_resolution: [4, 4, 45]
	src: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #ENC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_field_overrides: #FIELD_INFO_OVERRIDE
		// on_info_exists: "overwrite"
	}
	tmp_layer_dir: _
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_field_overrides: #FIELD_INFO_OVERRIDE
		// on_info_exists: "overwrite"
	}
	src_field: *null | {
		"@type": "build_cv_layer"
		path:    _
		data_resolution: _
		interpolation_mode: "field"
	}
}


#WARP_IMG_STAGE: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: [32, 32, 45]
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[256, 256, 0]]
	bbox: #BCUBE_COMBINED_32NM
	src: {
		"@type": "build_cv_layer"
		path: _
	}
	field: {
		"@type": "build_cv_layer"
		path: _
	}
	dst: {
		"@type": "build_cv_layer"
		path: _
		info_reference_path: src.path
	}
}

#ENCODE_STAGE: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseEncoder"
			model_path: #ENCODE_STAGES[0].model
		}
		crop_pad: [128, 128, 0]
	}
	dst_resolution: [32, 32, 45]
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[128, 128, 0]]
	bbox: #BCUBE_COMBINED_32NM
	src: {
		"@type": "build_cv_layer"
		path: _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_field_overrides: {
			data_type: "int8"
		}
		info_chunk_size:     [2048, 2048, 1]
		on_info_exists:      "overwrite"
	}
}


#MASK_IMG_STAGE: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "apply_mask_fn"
			"@mode": "partial"
		}
	}
	dst_resolution: [32, 32, 45]
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0]]
	bbox: #BCUBE_COMBINED_32NM
	src: {
		"@type": "build_cv_layer"
		path: _
	}
	dst: {
		"@type": "build_cv_layer"
		path: _
		info_reference_path: src.path
	}
	masks: [
		{
			"@type":             "build_cv_layer"
			path:                _
			read_procs: [
				{
					"@type":    "lambda"
					lambda_str: "lambda data: (data == 0)"
				},
			]
		},
	]
}


#JOINT_OFFSET_FLOW: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230405"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      100
	batch_gap_sleep_sec:  1
	do_dryrun_estimation: true
	local_test:           false

	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for z_offset in #Z_OFFSETS {
				"@type": "mazepa.concurrent_flow"
				stages: [
					{
						"@type": "mazepa.seq_flow"
						stages: [
							// Fine Alignment - Good coarse
							#CF_FLOW_TMPL & {
								dst: path:     "\(#FIELDS_PATH)/fine/\(z_offset)"
								tmp_layer_dir: "\(#FIELDS_PATH)/fine/\(z_offset)/tmp"
								tgt_offset: [0, 0, z_offset]
								stages: #ALIGN_STAGES
								src_field: {
									path:    "\(#FIELDS_PATH)/coarse/\(z_offset)"
									data_resolution: [256, 256, 45]
								}
							},

							// Warp good fine alignment
							#WARP_IMG_STAGE & {
								src: path:   #IMG_PATH
								field: path: "\(#FIELDS_PATH)/fine/\(z_offset)"
								dst: path:   "\(#OUTPUT_IMG_PATH)/fine/\(z_offset)"
								bbox: #BCUBE_COMBINED_32NM
							},

							#ENCODE_STAGE & {
								src: path: "\(#OUTPUT_IMG_PATH)/fine/\(z_offset)"
								dst: path: "\(#OUTPUT_ENC_PATH)/fine/\(z_offset)"
								bbox: #BCUBE_COMBINED_32NM
							},

							#MASK_IMG_STAGE & {
								src: path: "\(#OUTPUT_IMG_PATH)/fine/\(z_offset)"
								dst: path: "\(#OUTPUT_IMG_PATH)/fine_masked/\(z_offset)"
								masks: [
									{
										path: "\(#OUTPUT_ENC_PATH)/fine/\(z_offset)"
									}
								]
								bbox: #BCUBE_COMBINED_32NM
							},

						]
					},
				]
			},
		]
	}
}

[

	//ALIGN
	#JOINT_OFFSET_FLOW
	
]