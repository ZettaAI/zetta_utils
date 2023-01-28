#EXP_NAME:      "base_encodings"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"
#POST_WEIGHT:   1.85
#ZCONS_WEIGHT:  0.0
#LR:            1e-3
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   1.0
#EXP_VERSION:   "tmp_patch\(#CHUNK_XY)_post\(#POST_WEIGHT)_overfit_lr\(#LR)_deep_k\(#K)_clip\(#CLIP)_equi\(#EQUI_WEIGHT)_f1f2_x11"
#CHUNK_XY:      1024

//#MODEL_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#MODEL_CKPT: null // Set to a path to only load the net weights

#SRC_CV: "gs://sergiy_exp/pairs_dsets/zfish_x0/src"
#TGT_CV: "gs://sergiy_exp/pairs_dsets/zfish_x0/dst"

#FIELD_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 3
worker_image:   "us.gcr.io/zetta-research/zetta_utils:all_x11"
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
		"@type":                "BaseEncoder"
		field_magn_thr:         0.8
		val_log_row_interval:   1
		train_log_row_interval: 50
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
						kernel_sizes: #K
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
		val_check_interval:      null
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
						read_procs: [
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
					}
					tgt: {
						"@type": "build_cv_layer"
						path:    #TGT_CV
						cv_kwargs: {
							//cache: "/home/sergiy/.cloudvolume/memcache"
						}
						read_procs: [
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
					}
				}
			}
			sample_indexer: {
				"@type": "LoopIndexer"
				inner_indexer: {
					"@type": "VolumetricNGLIndexer"
					resolution: [32, 32, 30]
					desired_resolution: [32, 32, 30]
					chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
					path: "sergiy/base_training_overfit_x0"
				}
				desired_num_samples: 10000
			}
		}
	}
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
}
