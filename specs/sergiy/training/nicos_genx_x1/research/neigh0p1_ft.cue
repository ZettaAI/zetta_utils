#EXP_NAME:        "coarsener_gen_x1"
#TRAINING_ROOT:   "gs://sergiy_exp/training_artifacts/nicos_gen_x1"
#LR:              3e-4
#CLIP:            0e-5
#K:               3
#SIGNI_WEIGHT:    0.5
#FIELD_MAG:       0.8
#NEIGHBOR_WEIGHT: 0.1
#LOWCONF_WEIGHT:  0.0
#ENC_CH:          1
#EXP_VERSION:     "128_256_chunk_z\(#IMG_CHUNK_Z)_sig\(#SIGNI_WEIGHT)_lr\(#LR)_fieldmag\(#FIELD_MAG)_neighbor\(#NEIGHBOR_WEIGHT)_conv3_cns_1chlowconf\(#LOWCONF_WEIGHT)_cross_encch\(#ENC_CH)_x6_ft_x0"
#CHUNK_XY:        1024

#IMG_CHUNK_Z: 2 // 1 for same section warping, 2 for comparing dec(src) to tgt

#START_EXP_VERSION: "128_256_chunk_z2_sig0.5_lr0.0003_fieldmag0.8_neighbor0.0_conv3_cns_1chlowconf0.0_cross_encch1_x6"
#MODEL_CKPT:        "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#START_EXP_VERSION)/last.ckpt"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x49"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas: 1
local_test:      true

target: {
	"@type": "lightning_train"
	"@mode": "partial"
	regime: {
		"@type":                "EncodingCoarsenerGenX1Regime"
		field_magn_thr:         #FIELD_MAG
		significance_weight:    #SIGNI_WEIGHT
		lowconf_weight:         #LOWCONF_WEIGHT
		neighbor_weight:        #NEIGHBOR_WEIGHT
		val_log_row_interval:   16
		train_log_row_interval: 400
		lr:                     #LR
		encoder: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "ConvBlock"
						num_channels: [1, 32, 32, 32]
						activate_last: true
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
					},
					{
						"@type":     "torch.nn.AvgPool2d"
						kernel_size: 2
					},
					{
						"@type": "ConvBlock"
						num_channels: [32, 32, 32, #ENC_CH]
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
				"encoder",
			]
		}
		decoder: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "ConvBlock"
						num_channels: [#ENC_CH, 32, 32, 32]
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
						activate_last: true
					},
					{
						"@type":      "torch.nn.Upsample"
						mode:         "bilinear"
						scale_factor: 2
					},
					{
						"@type": "ConvBlock"
						num_channels: [32, 32, 32, 1]
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
				"decoder",
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
		val_check_interval:      600
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
		"pattern": "1 x y z -> z x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   127.0
	},

]

#dset_settings: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		field: #FIELD_DSET
		src: {
			"@type": "LayerDataset"
			layer: {
				"@type":    "build_cv_layer"
				path:       "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns_128nm_unet"
				read_procs: #IMG_PROCS
			}
			sample_indexer: {
				"@type": "VolumetricStridedIndexer"
				resolution: [128, 128, 45]
				desired_resolution: [128, 128, 45]
				chunk_size: [#CHUNK_XY, #CHUNK_XY, #IMG_CHUNK_Z]
				stride: [#CHUNK_XY, #CHUNK_XY, 1]
				bbox: {
					"@type":       "BBox3D.from_coords"
					"start_coord": _
					"end_coord":   _
					resolution:    _
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
		"resolution": [4, 4, 45]
		"start_coord": [0, 0, 0]
		"end_coord": [2048, 2048, 2040]
	}
	"stride": [256, 256, 1]
	"chunk_size": [#CHUNK_XY, #CHUNK_XY, 1]
	"resolution": [4, 4, 45]
	"desired_resolution": [4, 4, 45]
}

#train_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: #IMG_PROCS
	datasets: src: sample_indexer: bbox: {
		start_coord: [1024, 0, 2700]
		end_coord: [3072, 4096, 3100]
		resolution: [256, 256, 45]
	}
}

#val_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: #IMG_PROCS
	datasets: src: sample_indexer: bbox: {
		start_coord: [2048, 0, 3180]
		end_coord: [3072, 1024, 3188]
		resolution: [256, 256, 45]
	}
}
