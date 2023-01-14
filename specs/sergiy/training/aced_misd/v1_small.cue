#EXP_NAME:       "aced_misd"
#TRAINING_ROOT:  "gs://sergiy_exp/training_artifacts"
#LR:             1e-4
#CLIP:           0e-5
#K:              3
#CHUNK_XY:       1024
#FIELD_MAGN_THR: 1.0

#EXP_VERSION: "zm1_zm2_thr\(#FIELD_MAGN_THR)_scratch_large_skip3008_x1"

#MODEL_CKPT: null
//#MODEL_CKPT: "gs://sergiy_exp/training_artifacts/aced_misd/thr1.0_x1/last.ckpt"

#SRC_ZM1_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x1/src_zm1"
#SRC_ZM2_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x1/src_zm2"
#TGT_PATH:     "gs://sergiy_exp/pairs_dsets/zfish_x1/tgt"

#FIELD_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 3
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_all_x14"
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
	"@mode": "lazy"

	regime: {
		"@type":                "MisalignmentDetectorAcedRegime"
		field_magn_thr:         #FIELD_MAGN_THR
		val_log_row_interval:   10
		train_log_row_interval: 120
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
							[32, 32, 1],
						]
						kernel_sizes: #K
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
		val_check_interval:      480
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
		"@type": "add"
		"@mode": "partial"
		value:   -127.0
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   255.0
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
				read_postprocs: [
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
				"zm2": #IMG_DSET_TMPL & {
					layer: layers: {
						src: path: #SRC_ZM2_PATH
						tgt: path: #TGT_PATH
					}
				}
				"zm1": #IMG_DSET_TMPL & {
					layer: layers: {
						src: path: #SRC_ZM1_PATH
						tgt: path: #TGT_PATH
					}
				}
			}
		}
	}
}

#IMG_DSET_TMPL: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_layer_set"
		layers: {
			src: {
				"@type":        "build_cv_layer"
				path:           _
				read_postprocs: #IMG_PROCS
			}
			tgt: {
				"@type":        "build_cv_layer"
				path:           _
				read_postprocs: #IMG_PROCS
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [32, 32, 30]
		desired_resolution: [32, 32, 30]
		stride: [#CHUNK_XY, #CHUNK_XY, 1]
		chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
		bcube: {
			"@type":     "BoundingCube"
			start_coord: _
			end_coord:   _
			resolution: [
				4,
				4,
				30,
			]
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
#TRAIN_BCUBE: {
	"@type": "BoundingCube"
	resolution: [4, 4, 30]
	start_coord: [1024 * 35, 1024 * 65, 3002]
	end_coord: [1024 * 70, 1024 * 110, 3008]
}

#train_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	datasets: images: datasets: zm2: sample_indexer: bcube: #TRAIN_BCUBE
	datasets: images: datasets: zm1: sample_indexer: bcube: #TRAIN_BCUBE
}

#val_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}

	datasets: images: datasets: zm2: sample_indexer: bcube: {
		start_coord: [1024 * 35, 1024 * 65, 3014]
		end_coord: [1024 * 70, 1024 * 110, 3015]
	}
	datasets: images: datasets: zm1: sample_indexer: bcube: {
		start_coord: [1024 * 35, 1024 * 65, 3014]
		end_coord: [1024 * 70, 1024 * 110, 3015]
	}
}
