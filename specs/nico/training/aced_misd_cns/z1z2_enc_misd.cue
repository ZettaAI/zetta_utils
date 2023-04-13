#EXP_NAME:       "aced_misd_cns"
#TRAINING_ROOT:  "gs://zetta-research-nico/training_artifacts"
#LR:             1e-5
#CLIP:           0e-5
#K:              3
#CHUNK_XY:       1024
#FIELD_MAGN_THR: 5.0


#EXP_VERSION: "thr\(#FIELD_MAGN_THR)_lr\(#LR)_z1z2_400-500_2910-2920_more_aligned_unet5_32_finetune_2"
#MODEL_CKPT: "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr5.0_lr0.00005_z1z2_400-500_2910-2920_more_aligned_unet5_32/last.ckpt"

// #TGT_CV: "gs://zetta-research-nico/pairs_dsets/cns_x0_400-500/encs_warped/0"
// #SRC_Z2_PREFIX: "gs://zetta-research-nico/misd/enc/local_optima_400-500/enc_z2/med_7.5px_max_"
// #DISP_Z2_PREFIX: "gs://zetta-research-nico/misd/cns/local_optima_400-500/vec_length10x_z2/med_7.5px_max_"
// #MAX_DISP: 20

"@type":        "mazepa.execute_on_gcp_with_sqs"
worker_image:   "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230405"
worker_resources: {
	memory:           "38560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     1

local_test: false

#UNET_DOWNSAMPLE: {
	"@type": "torch.nn.MaxPool2d"
	"@mode": "partial"
	kernel_size: 2
}

#UNET_UPSAMPLE: {
	{
		"@type": "UpConv"
		"@mode": "partial"
		kernel_size: #K
		upsampler: {
			"@type": "torch.nn.Upsample"
			"@mode": "partial"
			scale_factor: 2
			mode: "nearest"
			align_corners: null
		},
		conv: {
			"@type": "torch.nn.Conv2d"
			"@mode": "partial"
			padding: "same"
		}
	}
}

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "MisalignmentDetectorAcedRegime"
		output_mode:            "binary"
		encoder_path:           null
		max_shared_displacement_px: 0.0
		max_src_displacement_px: {
			"@type": "uniform_distr"
			low:     0.0
			high:    0.0
		}
		equivar_rot_deg_distr: {
			"@type": "uniform_distr"
			low:     0.0
			high:    0.0
		}
		equivar_trans_px_distr: {
			"@type": "uniform_distr"
			low:     0.0
			high:    0.0
		}

		field_magn_thr:         #FIELD_MAGN_THR
		val_log_row_interval:   4
		train_log_row_interval: 200
		lr:                     #LR
		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "UNet"
						"@version": "0.0.2"
						list_num_channels: [
							[2, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],

							[32, 32, 32],

							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
						]
						downsample: #UNET_DOWNSAMPLE
						upsample: #UNET_UPSAMPLE
						activate_last: true
						kernel_sizes: [#K, #K]
						padding_modes: "zeros"
						unet_skip_mode: "sum"
						skips: {"1": 2}
					},
					{
						"@type": "torch.nn.Conv2d"
						in_channels: 32
						out_channels: 1
						kernel_size: 1
					},
					{
						"@type": "torch.nn.Sigmoid"
					}
				]
			},
			ckpt_path: #MODEL_CKPT
			component_names: [
				"model",
			]
		}
	}
	trainer: {
		"@type":                 "ZettaDefaultTrainer"
		accelerator:             "gpu"
		devices:                 1
		max_epochs:              100
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      1000
		gradient_clip_algorithm: "norm"
		gradient_clip_val:       #CLIP
		checkpointing_kwargs: {
			update_every_n_secs: 60
			backup_every_n_secs: 900
		}
	}

	train_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  8
		shuffle:     true
		num_workers: 12
		dataset:     #TRAINING_DSET
	}
	val_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  4
		shuffle:     false
		num_workers: 8
		dataset:     #VAL_DSET
	}
}


#IMG_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   127.0
	},
]

#DISP_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   10.0
	},
]


#TRAINING_DSET: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		images: {
			"@type": "JointDataset"
			mode:    "vertical"
			datasets: {
				for z_offset in [1, 2] {
					"z400_500_\(z_offset)": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_400-500/fine_misaligned/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/encodings_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_400-500/fine_diff3/-\(z_offset)"
									read_procs: #DISP_PROCS
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "ChainIndexer"
								inner_indexer: [
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [1 * 2048, 1 * 2048, 400]
											end_coord: [4 * 2048, 4 * 2048, 498]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [6 * 2048, 1 * 2048, 400]
											end_coord: [9 * 2048, 4 * 2048, 498]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [11 * 2048, 1 * 2048, 400]
											end_coord: [15 * 2048, 4 * 2048, 498]
											resolution: [32, 32, 45]
										}
									}
								]
							}
						}
					},
					"z400_500_\(z_offset)_aligned": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_400-500/fine/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/encodings_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "file:///tmp/placeholder_400-500"
									cv_kwargs: {
										fill_missing: true
									}
									info_reference_path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_400-500/fine_diff3/-\(z_offset)"
									read_procs: [
										{
											"@mode":   "partial"
											"@type":   "rearrange"
											"pattern": "c x y 1 -> c x y"
										},
										{
											"@type": "torch.zeros_like"
											"@mode": "partial"
										},
										{
											"@type": "torch.add"
											"@mode": "partial"
											other: 0.0
										}
									]
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "ChainIndexer"
								inner_indexer: [
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [1 * 2048, 1 * 2048, 400]
											end_coord: [4 * 2048, 4 * 2048, 498]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [6 * 2048, 1 * 2048, 400]
											end_coord: [9 * 2048, 4 * 2048, 498]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [11 * 2048, 1 * 2048, 400]
											end_coord: [15 * 2048, 4 * 2048, 498]
											resolution: [32, 32, 45]
										}
									}
								]
							}
						}
					},
					"z2910_2920_\(z_offset)": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_2908-2921/fine_misaligned/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/pairs_dsets/cns_x0_2910-2920_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_2908-2921/fine_diff3/-\(z_offset)"
									read_procs: #DISP_PROCS
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "ChainIndexer"
								inner_indexer: [
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [3 * 1024, 2 * 1024, 2910]
											end_coord: [27 * 1024, 8 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [13 * 1024, 8 * 1024, 2910]
											end_coord: [16 * 1024, 16 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [12 * 1024, 16 * 1024, 2910]
											end_coord: [21 * 1024, 20 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [12 * 1024, 21 * 1024, 2910]
											end_coord: [17 * 1024, 25 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									}
								]
							}
						}
					},
					"z2910_2920_\(z_offset)_aligned": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_2908-2921/fine/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/pairs_dsets/cns_x0_2910-2920_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "file:///tmp/placeholder_2908-2921"
									cv_kwargs: {
										fill_missing: true
									}
									info_reference_path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_2908-2921/fine_diff3/-\(z_offset)"
									read_procs: [
										{
											"@mode":   "partial"
											"@type":   "rearrange"
											"pattern": "c x y 1 -> c x y"
										},
										{
											"@type": "torch.zeros_like"
											"@mode": "partial"
										},
										{
											"@type": "torch.add"
											"@mode": "partial"
											other: 0.0
										}
									]
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "ChainIndexer"
								inner_indexer: [
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [3 * 1024, 2 * 1024, 2910]
											end_coord: [27 * 1024, 8 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [13 * 1024, 8 * 1024, 2910]
											end_coord: [16 * 1024, 16 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [12 * 1024, 16 * 1024, 2910]
											end_coord: [21 * 1024, 20 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									},
									{
										"@type": "VolumetricStridedIndexer"
										resolution: [32, 32, 45]
										stride: [#CHUNK_XY, #CHUNK_XY, 1]
										chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
										bbox: {
											"@type":     "BBox3D.from_coords"
											start_coord: [12 * 1024, 21 * 1024, 2910]
											end_coord: [17 * 1024, 25 * 1024, 2921]
											resolution: [32, 32, 45]
										}
									}
								]
							}
						}
					},
					"false_neg_z\(z_offset)": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_3406-3410/fine/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/pairs_dsets/cns_x0_3406-3410_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "file:///tmp/placeholder_3406-3410"
									cv_kwargs: {
										fill_missing: true
									}
									info_reference_path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_2908-2921/fine_diff3/-\(z_offset)"
									read_procs: [
										{
											"@mode":   "partial"
											"@type":   "rearrange"
											"pattern": "c x y 1 -> c x y"
										},
										{
											"@type": "torch.full_like"
											"@mode": "partial"
											fill_value: 255.0
										},
										{
											"@type": "torch.add"
											"@mode": "partial"
											other: 0.0
										}
									]
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "LoopIndexer"
								if z_offset == 1 {
									desired_num_samples: 12500
								}
								if z_offset == 2 {
									desired_num_samples: 8000
								}
								inner_indexer: {
									"@type": "VolumetricNGLIndexer"
									resolution: [32, 32, 45]
									chunk_size: [1024, 1024, 1]
									path: "nkem/cns/false_neg_z\(z_offset)"
								}
							}
						}
					}
				},
			}
		}
	}
}


#VAL_DSET: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		images: {
			"@type": "JointDataset"
			mode:    "vertical"
			datasets: {
				for z_offset in [2] {
					"z2000_2001_\(z_offset)": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_1998-2001/fine_misaligned/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/pairs_dsets/cns_x0_1998-2001_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_1998-2001/fine_diff3/-\(z_offset)"
									read_procs: #DISP_PROCS
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "VolumetricStridedIndexer"
								resolution: [32, 32, 45]
								stride: [#CHUNK_XY, #CHUNK_XY, 1]
								chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
								bbox: {
									"@type":     "BBox3D.from_coords"
									start_coord: [3 * 1024, 3 * 1024, 2000]
									end_coord: [14 * 1024, 7 * 1024, 2001]
									resolution: [32, 32, 45]
								}
							},
						}
					},
					"z2000_2001_\(z_offset)_aligned": {
						"@type": "LayerDataset"
						layer: {
							"@type": "build_layer_set"
							layers: {
								src: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/misd/cns/pairwise_enc_1998-2001/fine/-\(z_offset)"
									read_procs: #IMG_PROCS
								}
								tgt: {
									"@type": "build_cv_layer"
									path:    "gs://zetta-research-nico/pairs_dsets/cns_x0_1998-2001_masked"
									read_procs: #IMG_PROCS
									index_procs: [
										{
											"@type": "VolumetricIndexTranslator"
											offset: [0, 0, -z_offset]
											resolution: [32, 32, 45]
										}
									]
								}
								displacement: {
									"@type": "build_cv_layer"
									path:    "file:///tmp/placeholder_1998-2001"
									cv_kwargs: {
										fill_missing: true
									}
									info_reference_path:    "gs://zetta-research-nico/misd/cns/pairwise_fields_1998-2001/fine_diff3/-\(z_offset)"
									read_procs: [
										{
											"@mode":   "partial"
											"@type":   "rearrange"
											"pattern": "c x y 1 -> c x y"
										},
										{
											"@type": "torch.zeros_like"
											"@mode": "partial"
										},
										{
											"@type": "torch.add"
											"@mode": "partial"
											other: 0.0
										}
									]
								}
							}
						}
						sample_indexer: {
							"@type": "RandomIndexer"
							inner_indexer: {
								"@type": "VolumetricStridedIndexer"
								resolution: [32, 32, 45]
								stride: [#CHUNK_XY, #CHUNK_XY, 1]
								chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
								bbox: {
									"@type":     "BBox3D.from_coords"
									start_coord: [3 * 1024, 3 * 1024, 2000]
									end_coord: [14 * 1024, 7 * 1024, 2001]
									resolution: [32, 32, 45]
								}
							},
						}
					},
				}
			}
		}
	}
}
