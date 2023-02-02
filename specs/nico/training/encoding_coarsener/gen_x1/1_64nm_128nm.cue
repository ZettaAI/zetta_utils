#EXP_NAME:      "coarsener_gen_x1"
#TRAINING_ROOT: "gs://zetta-research-nico/training_artifacts"
#LR:            3e-4
#CLIP:          0e-5
#K:             3
#SIGNI_WEIGHT:  0.2
#FIELD_MAG:     16.0
#EXP_VERSION:   "64_128_sig\(#SIGNI_WEIGHT)_lr\(#LR)_fieldmag\(#FIELD_MAG)_zfish_cns_2"
#CHUNK_XY:      1024

#MODEL_CKPT: null

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
		"@type":                "EncodingCoarsenerGenX1Regime"
		field_magn_thr:         #FIELD_MAG
		significance_weight:    #SIGNI_WEIGHT
		val_log_row_interval:   16
		train_log_row_interval: 256
		lr:                     #LR
		encoder: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "ConvBlock"
						num_channels: [1, 32, 32]
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
		decoder: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "ConvBlock"
						num_channels: [1, 32, 32]
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
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   127
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
				path:       "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns_64nm"
				read_procs: #IMG_PROCS
			}
			sample_indexer: {
				"@type": "VolumetricStridedIndexer"
				resolution: [64, 64, 45]
				desired_resolution: [64, 64, 45]
				chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
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
		"end_coord": [2048, 2048, 2040]
		"resolution": [4, 4, 45]
		"start_coord": [0, 0, 0]
	}
	"stride": [64, 64, 1]
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
		start_coord: [1280, 0, 2900]
		end_coord: [2816, 3584, 3000]
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
		start_coord: [1280, 0, 3180]
		end_coord: [2816, 3584, 3181]
		resolution: [256, 256, 45]
	}
}
