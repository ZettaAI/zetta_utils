#EXP_NAME:      "encoding_coarsener"
#EXP_VERSION:   "x0_arti"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"

// #FULL_STATE_CKPT wile load the WHOLE TRAINING STATE.
// This will resume training from the checkpoint AND DISREGARD OTHER
// PARAMETERS IN THIS FILE, such as new learning rates etc.
//#FULL_STATE_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#FULL_STATE_CKPT: null
#ENC_CV:          "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

#ENCODER_ARCH: {
	"@type": "ArtificerySpec"
	spec: {
		type: "TransparentSeq"
		modules: [
			{
				type: "convblock"
				arch_desc: {
					fms: [1, 32, 32, 32]
					k: 3
				}
			},
			{
				type: "custom_layer"
				"import": ["torch"]
				"code": "result = torch.nn.AvgPool2d(2, count_include_pad=False)"
			},
			{
				type: "convblock"
				arch_desc: {
					fms: [32, 32, 32, 1]
					k: 3
				}
			},
		]
	}
}

#DECODER_ARCH: {
	"@type": "ArtificerySpec"
	spec: {
		type: "TransparentSeq"
		modules: [
			{
				type: "convblock"
				arch_desc: {
					fms: [1, 32, 32, 32]
					k: 3
				}
			},
			{
				type: "interpolate"
			},
			{
				type: "convblock"
				arch_desc: {
					fms: [32, 32, 32, 1]
					k: 3
				}
			},
		]
	}
}

//dset specs
#dset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "LayerSet"
		layers: {
			data_in: {
				"@type": "CVLayer"
				path:    #ENC_CV
				//cv_kwargs: {cache: true}
				read_postprocs: [
					{
						"@type": "Squeeze"
						dim:     -1
					},
				]
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStepIndexer"
		desired_resolution: [64, 64, 40]
		index_resolution: [64, 64, 40]
		patch_size_resolution: [64, 64, 40]
		patch_size: [1024, 1024, 1]
		step_size: [512, 512, 1]
		step_size_resolution: [64, 64, 40]
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
			end_coord: [230000, 80000, 2099]
			resolution: [4, 4, 40]
		}
	}
}

#val_dset: #dset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [80000, 30000, 2099]
			end_coord: [230000, 80000, 2100]
			resolution: [4, 4, 40]
		}
	}
}
"@type": "lightning_train"

ckpt_path: #FULL_STATE_CKPT
regime: {
	"@type": "EncodingCoarsener"
	lr:      4e-4
	encoder: #ENCODER_ARCH
	decoder: #DECODER_ARCH
	// model_ckpt_path: #LAST_CKPT_PATH
}
trainer: {
	"@type":            "ZettaDefaultTrainer"
	accelerator:        "gpu"
	devices:            1
	max_epochs:         100
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  500
	val_check_interval: 500
	checkpointing_kwargs: {
		//update_every_n_secs: 20
		// backup_every_n_secs: 900
	}
	profiler: "simple"
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
