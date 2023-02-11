#EXP_NAME:      "base_coarsener_cns"
#TRAINING_ROOT: "gs://zetta-research-nico/training_artifacts"
#POST_WEIGHT:   1.3
#FIELD_MAGN_THR: 0.8
#LR:            1e-4
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   0.5
#CHUNK_XY:      1024
#GAMMA_LOW:     0.5
#GAMMA_HIGH:    1.5
#GAMMA_PROB:    1.0
#TILE_LOW:      0.0
#TILE_HIGH:     0.2

#SOURCE_RES:    [32, 32, 45]
#SOURCE_PATH:   "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"

#EXP_VERSION:   "3_M3_M6_unet1_conv3_lr\(#LR)_equi\(#EQUI_WEIGHT)_post\(#POST_WEIGHT)_fmt\(#FIELD_MAGN_THR)_cns_all"

#START_EXP_VERSION: "2_M3_M6_unet1_conv3_lr0.0001_equi0.5_post1.5_fmt\(#FIELD_MAGN_THR)_cns_all"
#MODEL_CKPT:    "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/\(#START_EXP_VERSION)/last.ckpt"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230210"
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
		"@type":                "BaseEncoderCoarsenerRegime"
		field_magn_thr:         #FIELD_MAGN_THR
		max_displacement_px:    32.0
		val_log_row_interval:   24
		train_log_row_interval: 250
		lr:                     #LR
		equivar_weight:         #EQUI_WEIGHT
		post_weight:            #POST_WEIGHT
		ds_factor:              8
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
						]
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
					},
					{
						"@type":     "torch.nn.AvgPool2d"
						kernel_size: 2
					},
					{
						"@type": "ConvBlock"
						num_channels: [32, 32, 32]
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
						activate_last: true
					},
					{
						"@type":     "torch.nn.AvgPool2d"
						kernel_size: 2
					},
					{
						"@type": "ConvBlock"
						num_channels: [32, 32, 32]
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
						activate_last: true
					},
					{
						"@type":     "torch.nn.AvgPool2d"
						kernel_size: 2
					},
					{
						"@type": "ConvBlock"
						num_channels: [32, 32, 1]
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
						activate_last: false
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
		val_check_interval:      250
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
						path:       #SOURCE_PATH
						read_procs: _img_procs
					}
					tgt: {
						"@type":    "build_cv_layer"
						path:       #SOURCE_PATH
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
			resolution: #SOURCE_RES
			desired_resolution: #SOURCE_RES
			chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
			stride: [#CHUNK_XY, #CHUNK_XY, 1]
			bbox: {
				"@type":       "BBox3D.from_coords"
				start_coord: [10 * 1024, 1 * 1024, 2502]
				end_coord: [22 * 1024, 28 * 1024, 3500]
				resolution: [32, 32, 45]
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
		resolution: #SOURCE_RES
		desired_resolution: #SOURCE_RES
		chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
		stride: [#CHUNK_XY, #CHUNK_XY, 1]
		bbox: {
			"@type":       "BBox3D.from_coords"
			start_coord: [10 * 1024, 2 * 1024, 3182]
			end_coord: [22 * 1024, 7 * 1024, 3184]
			resolution: [32, 32, 45]
		}
	}
}
