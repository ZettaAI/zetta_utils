#EXP_NAME:         "coarsener_gen_x1"
#TRAINING_ROOT:    "gs://sergiy_exp/training_artifacts"
#LR:               3e-4
#CLIP:             0e-5
#K:                3
#SIGNI_WEIGHT:     0.5
#CENTERING_WEIGHT: 0.5
#EXP_VERSION:      "touch_up_x0_256nm_512nm_x6"
#CHUNK_XY:         1024

#MODEL_CKPT: null

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:??"
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
		field_magn_thr:         0.8
		significance_weight:    #SIGNI_WEIGHT
		centering_weight:       #CENTERING_WEIGHT
		val_log_row_interval:   8
		train_log_row_interval: 200
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
					},
					{
						"@type":     "torch.nn.AvgPool2d"
						kernel_size: 2
					},

					{
						"@type": "ConvBlock"
						num_channels: [32, 32, 1]
						kernel_sizes: [#K, #K]
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
		max_epochs:              1000
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      200
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
		value:   127
	},

]

#dset_settings: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		field: {
			"@type": "LayerDataset"
			layer: {
				"@type": "build_cv_layer"
				path:    "gs://zetta-research-nico/perlin_noise_fields/1px"
				cv_kwargs: {
					//cache: "/home/sergiy/.cloudvolume/memcache"
				}
				read_procs: [
					{
						"@type":   "rearrange"
						"@mode":   "partial"
						"pattern": "c x y 1 -> c x y"
					},
				]
			}
			"sample_indexer": #FIELD_INDEXER
		}
		src: {
			"@type": "LayerDataset"
			layer: {
				"@type":    "build_cv_layer"
				path:       "gs://zfish_unaligned/precoarse_x0/test_x0/encodings_x1"
				read_procs: #IMG_PROCS
			}
			sample_indexer: {
				"@type": "VolumetricStridedIndexer"
				resolution: [256, 256, 30]
				desired_resolution: [256, 256, 30]
				chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
				stride: [#CHUNK_XY / 4, #CHUNK_XY / 4, 1]
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

#train_dset: #dset_settings & {
	datasets: src: sample_indexer: bbox: {
		start_coord: [0, 0, 0]
		end_coord: [1024, 1024, 195]
		resolution: [512, 512, 30]

	}
}

#val_dset: #dset_settings & {
	datasets: src: sample_indexer: bbox: {
		start_coord: [0, 0, 198]
		end_coord: [1024, 1024, 200]
		resolution: [512, 512, 30]
	}
}

#FIELD_INDEXER: {
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
