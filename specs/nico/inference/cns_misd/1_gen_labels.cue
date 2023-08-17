import "math"
import "list"

// #BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced"
// #ENC_PATH: "\(#BASE_FOLDER)/coarse_x0/encodings_masked"
// #IMG_PATH: "\(#BASE_FOLDER)/coarse_x0/raw_img"


#ENC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/encodings_masked"
#IMG_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/raw_img"

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [32768, 36864, 7010]
	resolution: [32, 32, 45]
}

#BCUBE_COMBINED_32NM: {
	"@type":     "BBox3D.from_coords"
	start_coord: [0 * 2048, 0 * 2048, 6150]
	end_coord:   [16 * 2048, 16 * 2048, 6170]
	resolution:  [32, 32, 45]
}

#BCUBE_COMBINED_256NM: {
	let patch_x = #BCUBE_COMBINED_32NM.end_coord[0] div 2048
	let patch_y = #BCUBE_COMBINED_32NM.end_coord[1] div 2048
	let factor = 256 div 32
	"@type":     "BBox3D.from_coords"
	start_coord: [0 * 2048, 0 * 2048, 3405]
	end_coord:   [
		math.Ceil(patch_x / factor) * factor * 2048,
		math.Ceil(patch_y / factor) * factor * 2048,
		3406
	]
	resolution:  [32, 32, 45]
}

#DST_INFO_CHUNK_SIZE: [2048, 2048, 1]
#MEDIAN_DISP: 7.5
#MAX_DISP: 20
#Z_START: 6150
#Z_END: 6170


#INITIAL_PERLIN_FIELD_PATH: "gs://zetta-research-nico/misd/cns/initial_random_fields_256nm_\(#Z_START)-\(#Z_END)/med_\(#MEDIAN_DISP)px_max_\(#MAX_DISP)px"
#FIELDS_PATH: "gs://zetta-research-nico/misd/cns/pairwise_fields_\(#Z_START)-\(#Z_END)"
#OUTPUT_IMG_PATH: "gs://zetta-research-nico/misd/cns/pairwise_img_\(#Z_START)-\(#Z_END)"
#OUTPUT_ENC_PATH: "gs://zetta-research-nico/misd/cns/pairwise_enc_\(#Z_START)-\(#Z_END)"

#FIELD_INFO_OVERRIDE: {
	"data_type": "float32",
	"num_channels": 2,
	"scales": [
			for i in list.Range(0, 10, 1) {
			let ds_factor = [math.Pow(2, i), math.Pow(2, i), 1]
			let vx_res = [ for j in [1, 1, 2] {#DATASET_BOUNDS.resolution[j] * ds_factor[j]}]
			let ds_offset = [ for j in [1, 1, 2] {
				__div(#DATASET_BOUNDS.start_coord[j], ds_factor[j])
			}]
			let ds_size = [ for j in [1, 1, 2] {
				__div((#DATASET_BOUNDS.end_coord[j] - ds_offset[j]), ds_factor[j])
			}]

			chunk_sizes: [[ for j in [1, 1, 2] {list.Min([#DST_INFO_CHUNK_SIZE[j], ds_size[j]])}]]
			resolution:   vx_res
			encoding:     "zfpc"
			zfpc_correlated_dims: [true, true, false, false]
			zfpc_tolerance: 0.001953125
			key:          "\(vx_res[0])_\(vx_res[1])_\(vx_res[2])"
			voxel_offset: ds_offset
			size:         ds_size
		}
	],
	"type": "image"
}

#PREALIGN_STAGES: [
	#STAGE_TMPL & {
		dst_resolution: [2048, 2048, 45]
		fn: {
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		chunk_size: [576, 576, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       300
			num_iter: 1000
			lr:       0.015
		}
		chunk_size: [1152, 1152, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       300
			num_iter: 700
			lr:       0.015
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 45]

		fn: {
			sm:       50
			num_iter: 700
			lr:       0.03
		}
		chunk_size: [2048, 2048, 1]
	},
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
		dst_resolution: [1024, 1024, 45]
		fn: {
			sm:       10
			num_iter: 1000
			lr:       0.015
		}
		chunk_size: [1152, 1152, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [512, 512, 45]

		fn: {
			sm:       10
			num_iter: 700
			lr:       0.015
		}
		chunk_size: [2048, 2048, 1]
	},
	#STAGE_TMPL & {
		dst_resolution: [256, 256, 45]

		fn: {
			sm:       10
			num_iter: 700
			lr:       0.03
		}
		chunk_size: [2048, 2048, 1]
	},
]

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


#GEN_PERLIN_NOISE: {
	"@type": "mazepa.execute"
	target: {
		let bcube = #BCUBE_COMBINED_256NM
		let x_mult = (bcube.end_coord[0] - bcube.start_coord[0]) div 2048
		let y_mult = (bcube.end_coord[1] - bcube.start_coord[1]) div 2048
		"@type": "build_subchunkable_apply_flow"
		op: {
			"@type": "VolumetricCallableOperation"
			fn: {
				"@type": "gen_biased_perlin_noise_field"
				"@mode": "partial"
				shape: [2, x_mult * 2048 div 8, y_mult * 2048 div 8, 1]
				res:   [   x_mult * 2,    y_mult * 2      ]
				max_displacement_px: #MAX_DISP / 8.0
				field_magn_thr_px: #MEDIAN_DISP / 8.0
				octaves: 8
				device: "cuda"
			}
			crop_pad: [0, 0, 0]
		}
		dst_resolution: [256, 256, 45]
		processing_chunk_sizes: [[x_mult * 2048 div 8, y_mult * 2048 div 8, 1]]
		processing_crop_pads:   [[0, 0, 0]]
		bbox: bcube
		dst: {
			"@type":              "build_cv_layer"
			path:                 #INITIAL_PERLIN_FIELD_PATH
			info_field_overrides: #FIELD_INFO_OVERRIDE
		}
	}
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


#WARP_FIELD_STAGE_256NM: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "field"
		crop_pad: [256, 256, 0]
	}
	dst_resolution: [256, 256, 45]
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[256, 256, 0]]
	bbox: #BCUBE_COMBINED_256NM
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


#Z_OFFSETS: [-1, -2]
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
						"@type": "mazepa.sequential_flow"
						stages: [
							// Coarse Alignment
							#CF_FLOW_TMPL & {
								dst: path:     "\(#FIELDS_PATH)/coarse/\(z_offset)"
								tmp_layer_dir: "\(#FIELDS_PATH)/coarse/\(z_offset)/tmp"
								tgt_offset: [0, 0, z_offset]
								stages: #PREALIGN_STAGES
								src_field: null
							},

							// Mix in Perlin noise to 256nm alignment
							#WARP_FIELD_STAGE_256NM & {
								src: path:   "\(#FIELDS_PATH)/coarse/\(z_offset)"
								field: path: #INITIAL_PERLIN_FIELD_PATH
								dst: {
									path:   "\(#FIELDS_PATH)/coarse_misaligned/\(z_offset)"
									info_reference_path: "\(#FIELDS_PATH)/coarse/\(z_offset)"
								}
							},

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

							// Fine Alignment - Bad coarse
							#CF_FLOW_TMPL & {
								dst: path:     "\(#FIELDS_PATH)/fine_misaligned/\(z_offset)"
								tmp_layer_dir: "\(#FIELDS_PATH)/fine_misaligned/\(z_offset)/tmp"
								tgt_offset: [0, 0, z_offset]
								stages: #ALIGN_STAGES
								src_field: {
									path:    "\(#FIELDS_PATH)/coarse_misaligned/\(z_offset)"
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

							// Warp bad fine alignment
							#WARP_IMG_STAGE & {
								src: path:   #IMG_PATH
								field: path: "\(#FIELDS_PATH)/fine_misaligned/\(z_offset)"
								dst: path:   "\(#OUTPUT_IMG_PATH)/fine_misaligned/\(z_offset)"
								bbox: #BCUBE_COMBINED_32NM
							},

							#ENCODE_STAGE & {
								src: path: "\(#OUTPUT_IMG_PATH)/fine/\(z_offset)"
								dst: path: "\(#OUTPUT_ENC_PATH)/fine/\(z_offset)"
								bbox: #BCUBE_COMBINED_32NM
							},

							#ENCODE_STAGE & {
								src: path: "\(#OUTPUT_IMG_PATH)/fine_misaligned/\(z_offset)"
								dst: path: "\(#OUTPUT_ENC_PATH)/fine_misaligned/\(z_offset)"
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

							#MASK_IMG_STAGE & {
								src: path: "\(#OUTPUT_IMG_PATH)/fine_misaligned/\(z_offset)"
								dst: path: "\(#OUTPUT_IMG_PATH)/fine_misaligned_masked/\(z_offset)"
								masks: [
									{
										path: "\(#OUTPUT_ENC_PATH)/fine/\(z_offset)"
									}
								]
								bbox: #BCUBE_COMBINED_32NM
							},


							// Extract diff between both fine alignments - store magnitude
							{
								"@type": "build_subchunkable_apply_flow"
								fn: {
									"@type": "torch.sub", "@mode": "partial"
								}
								processing_chunk_sizes: [[2048, 2048, 1]]
								processing_crop_pads: [[256, 256, 0]]
								dst_resolution: _ | *[32, 32, 45]
								bbox: #BCUBE_COMBINED_32NM
								input: {
									"@type": "build_cv_layer"
									path:    "\(#FIELDS_PATH)/fine_misaligned/\(z_offset)"
								}
								other: {
									"@type": "build_cv_layer"
									path:    "\(#FIELDS_PATH)/fine/\(z_offset)"
								}
								dst: {
									"@type":             "build_cv_layer"
									path:                "\(#FIELDS_PATH)/fine_diff3/\(z_offset)"
									info_reference_path: #ENC_PATH
									info_field_overrides: {
										data_type: "uint8"
									}
									on_info_exists:      "overwrite"
									write_procs: [
										{
											"@type":    "lambda"
											lambda_str: "lambda data: (data.norm(dim=0, keepdim=True)*10.0).round().clamp(0, 255).byte()"
										}
									]
								}
							},

						]
					},
				]
			},
		]
	}
}

[
	//GENERATE FIELD
	#GEN_PERLIN_NOISE,

	//ALIGN
	#JOINT_OFFSET_FLOW

]
