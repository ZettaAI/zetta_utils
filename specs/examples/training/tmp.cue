#EXP_NAME:      "mimic_encodings"
#EXP_VERSION:   "tmp_x340342342103123"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"

//#MODEL_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#MODEL_CKPT: null // Set to a path to only load the net weights

#IMG_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
#ENC_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type": "lightning_train"
regime: {
	"@type": "NoOpRegime"
}
trainer: {
	"@type":            "ZettaDefaultTrainer"
	accelerator:        "gpu"
	devices:            1
	max_epochs:         1
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  100
	val_check_interval: 100
	checkpointing_kwargs: {
		update_every_n_secs: 60
		backup_every_n_secs: 900
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
				//cv_kwargs: {cache: true}
				read_postprocs: [
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
				//cv_kwargs: {cache: true}
				read_postprocs: [
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
		resolution: [64, 64, 40]
		desired_resolution: [64, 64, 40]
		chunk_size: [256, 256, 20]
		stride: [256, 256, 1]
		bcube: {
			"@type":     "BoundingCube"
			start_coord: _
			end_coord:   _
			resolution: [4, 4, 40]
		}
	}
}

#train_dset: #dset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [80000, 30000, 2000]
			end_coord: [230000, 80000, 2080]
			resolution: [4, 4, 40]
		}
	}
}

#val_dset: #dset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [80000, 30000, 2080]
			end_coord: [230000, 80000, 2100]
			resolution: [4, 4, 40]
		}
	}
}
train_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  4
	shuffle:     true
	num_workers: 4
	dataset:     #train_dset
}
val_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  4
	shuffle:     false
	num_workers: 4
	dataset:     #val_dset
}
