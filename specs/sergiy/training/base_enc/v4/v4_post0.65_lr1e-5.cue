#EXP_NAME:      "base_encodings"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"
#POST_WEIGHT:   0.65
#ZCONS_WEIGHT:  0.0
#LR:            1e-5
#EXP_VERSION:   "tmp_post\(#POST_WEIGHT)_lr\(#LR)_equi_x9"

//#MODEL_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#MODEL_CKPT: null // Set to a path to only load the net weights

#SRC_CV: "gs://sergiy_exp/pairs_dsets/zfish_x0/src"
#TGT_CV: "gs://sergiy_exp/pairs_dsets/zfish_x0/dst"

#FIELD_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:all_x4"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     1
worker_lease_sec:    5
batch_gap_sleep_sec: 5

local_test: false

target: {
	"@type": "lightning_train"
	"@mode": "lazy"

	regime: {
		"@type":              "BaseEncoder"
		lr:                   #LR
		post_weight:          #POST_WEIGHT
		zero_conserve_weight: #ZCONS_WEIGHT
		field_magn_thr:       0.8
		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "ConvBlock"
						num_channels: [1, 32, 32, 32, 32, 1]
						kernel_sizes: 5
						skips: {"1": 3}
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
		accelerator:             "cpu"
		devices:                 1
		max_epochs:              100
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       150
		val_check_interval:      150
		gradient_clip_algorithm: "norm"
		gradient_clip_val:       0.001
		checkpointing_kwargs: {
			update_every_n_secs: 60
			backup_every_n_secs: 900
		}
	}

	train_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     true
		num_workers: 16
		dataset:     #train_dset
	}
	val_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     false
		num_workers: 16
		dataset:     #val_dset
	}
}

//dset specs
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
				read_postprocs: [
					{
						"@type":   "rearrange"
						"@mode":   "partial"
						"pattern": "c x y 1 -> c x y"
					},
					{
						"@type": "rescale_max_magnitude_aug"
						"@mode": "partial"
						max_magn_distr: {
							"@type": "uniform_distr"
							low:     0.5
							high:    8
						}
					},
				]
			}
			"sample_indexer": _
		}
		images: {
			"@type": "LayerDataset"
			layer: {
				"@type": "build_layer_set"
				layers: {
					src: {
						"@type": "build_cv_layer"
						path:    #SRC_CV
						cv_kwargs: {
							//cache: "/home/sergiy/.cloudvolume/memcache"
						}
						read_postprocs: [
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
								"@type": "add"
								"@mode": "partial"
								value:   -0.5
							},
						]
					}
					tgt: {
						"@type": "build_cv_layer"
						path:    #TGT_CV
						cv_kwargs: {
							//cache: "/home/sergiy/.cloudvolume/memcache"
						}
						read_postprocs: [
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
								"@type": "add"
								"@mode": "partial"
								value:   -0.5
							},
						]
					}
				}
			}
			sample_indexer: {
				"@type": "VolumetricStridedIndexer"
				resolution: [32, 32, 30]
				desired_resolution: [32, 32, 30]
				chunk_size: [1024, 1024, 1]
				stride: [512, 512, 1]
				bcube: {
					"@type":     "BoundingCube"
					start_coord: _
					end_coord:   _
					resolution:  _
				}
			}
		}
	}
}
#field_indexer: {
	"@type": "VolumetricStridedIndexer"
	"bcube": {
		"@type": "BoundingCube"
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
		1024,
		1024,
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
	datasets: images: sample_indexer: bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 24, 1024 * 23, 3000]
		end_coord: [1024 * 73, 1024 * 100, 3016]
		resolution: [4, 4, 30]
	}
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
}

#val_dset: #dset_settings & {
	datasets: images: sample_indexer: bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 30, 1024 * 40, 3016]
		end_coord: [1024 * 60, 1024 * 80, 3017]
		resolution: [4, 4, 30]
	}
	datasets: field: sample_indexer: #field_indexer
}
