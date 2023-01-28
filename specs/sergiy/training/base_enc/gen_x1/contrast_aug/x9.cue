#EXP_NAME:      "base_encodings"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"
#POST_WEIGHT:   1.55
#ZCONS_WEIGHT:  0.0
#LR:            1e-4
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   0.5
#CHUNK_XY:      1024
#GAMMA_LOW:     0.33
#GAMMA_HIGH:    3.0
#GAMMA_PROB:    1.0
#TILE_LOW:      0.1
#TILE_HIGH:     0.4
#EXP_VERSION:   "gamma_low\(#GAMMA_LOW)_high\(#GAMMA_HIGH)_prob\(#GAMMA_PROB)_tile_\(#TILE_LOW)_\(#TILE_HIGH)_x10"

#START_EXP_VERSION: "ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17"
#MODEL_CKPT:        "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#START_EXP_VERSION)/last.ckpt"

#SRC_CV: "gs://sergiy_exp/pairs_dsets/zfish_x0/src"
#TGT_CV: "gs://sergiy_exp/pairs_dsets/zfish_x0/dst"

#FIELD_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x13"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas: 1
local_test:      false

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "BaseEncoderRegime"
		field_magn_thr:         0.8
		val_log_row_interval:   1
		train_log_row_interval: 150
		lr:                     #LR
		equivar_weight:         #EQUI_WEIGHT
		post_weight:            #POST_WEIGHT
		zero_conserve_weight:   #ZCONS_WEIGHT
		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "UNet"
						list_num_channels: [
							[1, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],

							[32, 32, 32],

							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 1],
						]
						kernel_sizes: [#K, #K]
					},
					{
						"@type": "torch.nn.Tanh"
					},
				]
			}
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
		val_check_interval:      100
		gradient_clip_algorithm: "norm"
		gradient_clip_val:       #CLIP
		checkpointing_kwargs: {
			update_every_n_secs: 60
			backup_every_n_secs: 900
		}
	}

	train_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     true
		num_workers: 8
		dataset:     #train_dset
	}
	val_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     false
		num_workers: 8
		dataset:     #val_dset
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
		value:   255.0
	},
	{
		"@type": "gamma_contrast_aug"
		"@mode": "partial"
		prob:    #GAMMA_PROB
		gamma_distr: {
			"@type": "uniform_distr"
			low:     #GAMMA_LOW
			high:    #GAMMA_HIGH
		}
		max_magn: 1.0
	},
	{
		"@type": "square_tile_pattern_aug"
		"@mode": "partial"
		prob:    1.0
		tile_size: {
			"@type": "uniform_distr"
			low:     64
			high:    1024
		}
		tile_stride: {
			"@type": "uniform_distr"
			low:     64
			high:    1024
		}
		max_brightness_change: {
			"@type": "uniform_distr"
			low:     0.1
			high:    0.6
		}
		rotation_degree: {
			"@type": "uniform_distr"
			low:     0
			high:    45
		}
		preserve_data_val: 0.0
		repeats:           3
		device:            "cpu"
	},
]

#dset_settings: {
	"@type":    "JointDataset"
	mode:       "horizontal"
	_img_procs: _
	datasets: {
		field: #FIELD_DSET
		images: {
			"@type": "JointDataset"
			mode:    "vertical"
			datasets: {
				zfish: {
					"@type": "LayerDataset"
					layer: {
						"@type": "build_layer_set"
						layers: {
							src: {
								"@type":    "build_cv_layer"
								path:       #SRC_CV
								read_procs: _img_procs
							}
							tgt: {
								"@type":    "build_cv_layer"
								path:       #TGT_CV
								read_procs: _img_procs
							}
						}
					}
					sample_indexer: {
						"@type": "VolumetricNGLIndexer"
						resolution: [32, 32, 30]
						desired_resolution: [32, 32, 30]
						chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
						path: _
					}
				}
				cns: {
					"@type": "LayerDataset"
					layer: {
						"@type": "build_layer_set"
						layers: {
							src: {
								"@type":    "build_cv_layer"
								path:       "gs://zetta_lee_fly_cns_001_alignment_temp/cutouts/pairwise_field_3000_3010_rig20/img_warped"
								read_procs: _img_procs
							}
							tgt: {
								"@type":    "build_cv_layer"
								path:       "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
								read_procs: _img_procs
							}
						}
					}
					sample_indexer: {
						"@type": "VolumetricNGLIndexer"
						resolution: [32, 32, 45]
						desired_resolution: [32, 32, 45]
						chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
						path: _
					}
				}
			}
		}
	}
}

#FIELD_DSET: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_cv_layer"
		path:    "gs://zetta-research-nico/perlin_noise_fields/1px"
		read_procs: [
			{
				"@type":   "rearrange"
				"@mode":   "partial"
				"pattern": "c x y 1 -> c x y"
			},
		]
	}
	"sample_indexer": _
}

#field_indexer: {
	"@type": "VolumetricStridedIndexer"
	"bbox": {
		"@type": "BBox3D.from_coords"
		"end_coord": [
			2048,
			2048,
			2040,
		]
		"resolution": [
			4,
			4,
			45,
		]
		"start_coord": [
			0,
			0,
			2030,
		]
	}
	"stride": [
		64,
		64,
		1,
	]
	"chunk_size": [
		#CHUNK_XY,
		#CHUNK_XY,
		1,
	]
	"resolution": [
		4,
		4,
		45,
	]
	"desired_resolution": [
		4,
		4,
		45,
	]
}

#train_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: #IMG_PROCS
	datasets: images: datasets: zfish: sample_indexer: path: "sergiy/base_enc_train_x0"
	datasets: images: datasets: cns: sample_indexer: path:   "cns/rig20_training_samples"
}

#val_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: [
		{
			"@mode":   "partial"
			"@type":   "rearrange"
			"pattern": "c x y 1 -> c x y"
		},
		{
			"@type": "divide"
			"@mode": "partial"
			value:   255.0
		},
	]
	datasets: images: datasets: zfish: sample_indexer: path: "sergiy/base_enc_val_bright_x0"
	datasets: images: datasets: cns: sample_indexer: path:   "sergiy/empty_set_x0"
}
