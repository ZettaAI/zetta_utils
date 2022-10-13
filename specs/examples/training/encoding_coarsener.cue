#EXP_NAME:    "encoding_coarsener"
#EXP_VERSION: "demo_x0"

#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"

// #FULL_STATE_CKPT will load the WHOLE TRAINING STATE.
//#FULL_STATE_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"

//#FULL_STATE_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x1/last.ckpt"
#FULL_STATE_CKPT: null

//#ENCODER_CKPT:    "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x3/last.ckpt"
//#DECODER_CKPT:    "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x3/last.ckpt"
#ENCODER_CKPT: null
#DECODER_CKPT: null

#ENC_CV: "gs://fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m5_x0"

///////////////////////////////////////////////////////////////////
//////////////////////// Training Spec ////////////////////////////
///////////////////////////////////////////////////////////////////

"@type":   "lightning_train"
ckpt_path: #FULL_STATE_CKPT
regime: {
	"@type": "EncodingCoarsener"
	lr:      4e-4
	encoder: {
		"@type":   "load_weights_file"
		model:     #ENCODER_MODEL
		ckpt_path: #ENCODER_CKPT
		component_names: ["encoder"]
	}
	decoder: {
		"@type":   "load_weights_file"
		model:     #DECODER_MODEL
		ckpt_path: #DECODER_CKPT
		component_names: ["decoder"]
	}
	apply_counts: [1, 2, 3]
	invar_angle_range: [1, 180]
	invar_mse_weight: 0.1
	diffkeep_angle_range: [1, 5]
	diffkeep_weight: 0.1
}
trainer: {
	"@type":            "ZettaDefaultTrainer"
	accelerator:        "gpu"
	devices:            1
	max_epochs:         100
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  100
	val_check_interval: 100
	checkpointing_kwargs: {
		//update_every_n_secs: 20
		// backup_every_n_secs: 900
	}
	profiler: "simple"
}

///////////////////////////////////////////////////////////////////
////////////////////// Architecture Spec //////////////////////////
///////////////////////////////////////////////////////////////////
#ENCODER_MODEL: {
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

#DECODER_MODEL: {
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

///////////////////////////////////////////////////////////////////
///////////////////////// Dataset Spec ////////////////////////////
///////////////////////////////////////////////////////////////////
#dset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_layer_set"
		layers: {
			data_in: {
				"@type": "build_cv_layer"
				path:    #ENC_CV
				//cv_kwargs: {cache: true}
				read_postprocs: [
					{
						"@type": "rearrange"
						"@mode": "partial"
						pattern: "c x y 1 -> c x y"
					},
				]
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStepIndexer"
		desired_resolution: [128, 128, 40]
		index_resolution: [128, 128, 40]
		patch_size_resolution: [128, 128, 40]
		patch_size: [1024, 1024, 1]
		step_size: [512, 512, 1]
		step_size_resolution: [128, 128, 40]
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
