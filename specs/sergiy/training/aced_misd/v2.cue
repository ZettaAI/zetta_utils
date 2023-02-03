#EXP_NAME:       "aced_misd"
#TRAINING_ROOT:  "gs://sergiy_exp/training_artifacts"
#LR:             1e-4
#CLIP:           0e-5
#K:              3
#EXP_VERSION:    "v2_x1"
#CHUNK_XY:       1024
#FIELD_MAGN_THR: 0.8

#MODEL_CKPT: null

#FIELD_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 1000
worker_image:   "us.gcr.io/zetta-research/zetta_utils:all_x12"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     1
worker_lease_sec:    5
batch_gap_sleep_sec: 5

local_test: true

target: {
	"@type": "lightning_train"

	regime: {
		"@type":                "MisalignmentDetectorAcedRegime"
		field_magn_thr:         #FIELD_MAGN_THR
		val_log_row_interval:   1
		train_log_row_interval: 50
		lr:                     #LR
		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "UNet"
						list_num_channels: [
							[2, 32, 32],
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
		max_epochs:              100
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      350
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
			"@type": "JointDataset"
			mode:    "vertical"

			datasets: {
				zm1: {
					"@type": "LayerDataset"
					layer: {
						"@type": "build_layer_set"
						layers: {
							src: {
								"@type": "build_cv_layer"
								path:    "gs://zfish_unaligned/precoarse_x0/enc_gen3_x0"
								cv_kwargs: {
									//cache: "/home/sergiy/.cloudvolume/memcache"
								}
								read_procs: #IMG_PROCS
							}
							tgt: {
								"@type": "build_cv_layer"
								path:    "gs://zfish_unaligned/precoarse_x0/enc_gen3_x0/aligned_z-1"
								cv_kwargs: {
									//cache: "/home/sergiy/.cloudvolume/memcache"
								}
								read_procs: #IMG_PROCS
							}
						}
					}
					sample_indexer: {
						"@type": "VolumetricNGLIndexer"
						resolution: [32, 32, 30]
						desired_resolution: [32, 32, 30]
						chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
						path: "sergiy/base_enc_train_defects_x3"
					}
				}

				zm2: {
					"@type": "LayerDataset"
					layer: {
						"@type": "build_layer_set"
						layers: {
							src: {
								"@type": "build_cv_layer"
								path:    "gs://zfish_unaligned/precoarse_x0/enc_gen3_x0"
								cv_kwargs: {
									//cache: "/home/sergiy/.cloudvolume/memcache"
								}
								read_procs: #IMG_PROCS
							}
							tgt: {
								"@type": "build_cv_layer"
								path:    "gs://zfish_unaligned/precoarse_x0/enc_gen3_x0/aligned_z-2"
								cv_kwargs: {
									//cache: "/home/sergiy/.cloudvolume/memcache"
								}
								read_procs: #IMG_PROCS
							}
						}
					}
					sample_indexer: {
						"@type": "VolumetricNGLIndexer"
						resolution: [32, 32, 30]
						desired_resolution: [32, 32, 30]
						chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
						path: "sergiy/misd_train_zm2_x3"
					}
				}
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
