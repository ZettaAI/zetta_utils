#EXP_NAME:      "aff_tests"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"
#LR:            1e-5
#CLIP:          0e-5
#K:             3
#CHUNK_SIZE: [256, 256, 20]
#MODEL_CKPT:  null
#EXP_VERSION: "k\(#K)_lr\(#LR)_x8_no_ckpt_no_wandb"
"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x100_no_wandb"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     1
batch_gap_sleep_sec: 5

local_test: false

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "NaiveSupervisedRegime"
		val_log_row_interval:   1000000
		train_log_row_interval: 2000000
		lr:                     #LR

		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "UNet"
						conv: {
							"@type": "torch.nn.Conv3d"
							"@mode": "partial"
						}
						downsample: {
							"@type": "torch.nn.AvgPool3d"
							"@mode": "partial"
							kernel_size: [2, 2, 1]
						}
						upsample: {
							"@type": "torch.nn.Upsample"
							"@mode": "partial"
							scale_factor: [2, 2, 1]
							mode: "trilinear"
						}
						list_num_channels: [
							[1, 32, 32],
							[32, 32, 32],
							[32, 32, 32],

							[32, 32, 32],

							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 3],
						]
						kernel_sizes: [#K, #K, #K]
					},
					{
						"@type": "torch.nn.Sigmoid"
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
		max_epochs:              400
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      200
		gradient_clip_algorithm: "norm"
		gradient_clip_val:       #CLIP
		checkpointing_kwargs: {
			update_every_n_secs: 60000
			backup_every_n_secs: 900000
		}
	}

	train_dataloader: {
		"@type":            "TorchDataLoader"
		batch_size:         1
		shuffle:            true
		num_workers:        8
		dataset:            #train_dset
		persistent_workers: true
	}
	val_dataloader: {
		"@type":            "TorchDataLoader"
		batch_size:         1
		shuffle:            false
		num_workers:        4
		dataset:            #val_dset
		persistent_workers: true
	}
}

#dset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_layer_set"
		layers: {
			data_in: {
				"@type": "build_cv_layer"
				path:    "gs://sergiy_exp/aff_dsets/x0/img"
				cv_kwargs: {
					//cache: "/home/sergiy/.cloudvolume/memcache"
				}
				read_procs: [
					{"@type": "divide", "@mode": "partial", value: 127.5},
					{"@type": "add", "@mode":    "partial", value: -1},
				]

			}
			target: {
				"@type": "build_cv_layer"
				path:    "gs://sergiy_exp/aff_dsets/x0/aff_bin"
				cv_kwargs: {
					//cache: "/home/sergiy/.cloudvolume/memcache"
				}
				read_procs: [
					{"@type": "compare", "@mode":    "partial", mode: ">", value: 0},
					{"@type": "to_float32", "@mode": "partial"},
				]
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [8, 8, 45]
		chunk_size: #CHUNK_SIZE
		stride:     _
		bbox: {
			"@type":     "BBox3D.from_coords"
			start_coord: _
			end_coord:   _
			resolution: [
				4,
				4,
				45,
			]
		}
	}
}

#train_dset: #dset_settings & {
	sample_indexer: stride: [128, 128, 1]
	//sample_indexer: stride: #CHUNK_SIZE
	sample_indexer: bbox: {
		start_coord: [1024 * 200, 1024 * 80, 200]
		end_coord: [1024 * 205, 1024 * 85, 380]
	}
}

#val_dset: #dset_settings & {
	sample_indexer: stride: #CHUNK_SIZE
	sample_indexer: bbox: {
		start_coord: [1024 * 200, 1024 * 80, 380]
		end_coord: [1024 * 205, 1024 * 85, 400]
	}
}
