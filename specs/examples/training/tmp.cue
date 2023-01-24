#EXP_NAME:      "mimic_encodings"
#EXP_VERSION:   "tmp_x340342342180asfd031121233290123"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"

//#MODEL_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#MODEL_CKPT: null // Set to a path to only load the net weights

#IMG_CV: "gs://sergiy_exp/aff_dsets/x0/img"
#ENC_CV: "gs://sergiy_exp/aff_dsets/x0/aff"

"@type": "lightning_train"
regime: {
	"@type": "NoOpRegime"
}
trainer: {
	"@type":            "ZettaDefaultTrainer"
	accelerator:        "cpu"
	devices:            1
	max_epochs:         100
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  10000
	val_check_interval: null
	checkpointing_kwargs: {
		update_every_n_secs: 600
		backup_every_n_secs: 9000
	}
	profiler: "simple"
}
//dset specs
#dset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_layer_set"
		layers: {
			data_in: {
				"@type": "build_cv_layer"
				path:    #IMG_CV
				cv_kwargs: {
					parallel:  false
					cache:     true
					lru_bytes: 1024 * 1024 * 1024 * 2
				}
				read_procs: [
					{
						"@type": "divide"
						"@mode": "partial"
						value:   1.0
					},
				]
			}
			target: {
				"@type": "build_cv_layer"
				path:    #ENC_CV
				cv_kwargs: {
					parallel:  false
					cache:     true
					lru_bytes: 1024 * 1024 * 2
				}
				read_procs: [
					{
						"@type": "divide"
						"@mode": "partial"
						value:   1.0
					},

				]
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [8, 8, 45]
		desired_resolution: [8, 8, 45]
		chunk_size: [256 + 128, 256 + 128, 20]
		stride: [256 * 32, 256 * 32, 1]
		bbox: {
			"@type":     "BBox3D.from_coords"
			start_coord: _
			end_coord:   _
			resolution: [4, 4, 45]
		}
	}
}

#train_dset: #dset_settings & {
	sample_indexer: {
		bbox: {
			"@type": "BBox3D.from_coords"
			start_coord: [1024 * 200, 1024 * 80, 200]
			end_coord: [1024 * 205, 1024 * 85, 380]
			resolution: [4, 4, 45]
		}
	}
}

#val_dset: #dset_settings & {
	sample_indexer: {
		bbox: {
			"@type": "BBox3D.from_coords"
			start_coord: [1024 * 200, 1024 * 80, 380]
			end_coord: [1024 * 205, 1024 * 85, 400]
			resolution: [4, 4, 45]
		}
	}
}
train_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  4
	shuffle:     true
	num_workers: 0
	dataset:     #train_dset
}
val_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  4
	shuffle:     false
	num_workers: 0
	dataset:     #val_dset
}
