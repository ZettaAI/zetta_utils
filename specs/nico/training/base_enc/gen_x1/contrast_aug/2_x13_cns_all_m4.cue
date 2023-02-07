#EXP_NAME:      "base_encodings"
#TRAINING_ROOT: "gs://zetta-research-nico/training_artifacts"
#POST_WEIGHT:   1.4
#ZCONS_WEIGHT:  0.0
#LR:            1e-4
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   0.5
#CHUNK_XY:      1024
#GAMMA_LOW:     0.75
#GAMMA_HIGH:    1.5
#GAMMA_PROB:    1.0
#TILE_LOW:      0.0
#TILE_HIGH:     0.2
#EXP_VERSION:   "gamma_low\(#GAMMA_LOW)_high\(#GAMMA_HIGH)_prob\(#GAMMA_PROB)_tile_\(#TILE_LOW)_\(#TILE_HIGH)_lr\(#LR)_post\(#POST_WEIGHT)_cns_all_m4_3"

// #START_EXP_VERSION: "gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.0001_post1.8_cns_all"
#MODEL_CKPT:    "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.0001_post1.6_cns_all_m4_2/last.ckpt"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_3"
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
		max_displacement_px:    32.0
		val_log_row_interval:   8
		train_log_row_interval: 500
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
						padding_modes: "reflect"
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
		max_epochs:              1
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      500
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
		num_workers: 4
		dataset:     #train_dset
	}
	val_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     false
		num_workers: 4
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
			low:     #TILE_LOW
			high:    #TILE_HIGH
		}
		rotation_degree: {
			"@type": "uniform_distr"
			low:     0
			high:    90
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
			"@type": "LayerDataset"
			layer: {
				"@type": "build_layer_set"
				layers: {
					src: {
						"@type":    "build_cv_layer"
						path:       "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
						read_procs: _img_procs
					}
					tgt: {
						"@type":    "build_cv_layer"
						path:       "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
						read_procs: _img_procs
						index_procs: [
                            {
                                "@type": "VolumetricIndexTranslator"
                                offset: [0, 0, -1]
                                resolution: [4, 4, 45]
                            },
                        ]
					}
				}
			}
			"sample_indexer": _
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
			0,
		]
	}
	"stride": [
		1024,
		1024,
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
	datasets: images: sample_indexer: {
		"@type": "RandomIndexer"
		inner_indexer: {
			"@type": "VolumetricStridedIndexer"
			resolution: [64, 64, 45]
			desired_resolution: [64, 64, 45]
			chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
			stride: [#CHUNK_XY, #CHUNK_XY, 1]
			bbox: {
				"@type":       "BBox3D.from_coords"
				start_coord: [1280, 0, 2701]
				end_coord: [2816, 3584, 3180]
				resolution: [256, 256, 45]
			}
		}
	}
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
	datasets: images: sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [64, 64, 45]
		desired_resolution: [64, 64, 45]
		chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
		stride: [#CHUNK_XY, #CHUNK_XY, 1]
		bbox: {
			"@type":       "BBox3D.from_coords"
			start_coord: [2048, 256, 3180]
			end_coord: [2512, 1024, 3182]
			resolution: [256, 256, 45]
		}
	}
}
