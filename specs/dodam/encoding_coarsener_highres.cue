#EXP_NAME:    "encoding_coarsener_sanity2"
#EXP_VERSION: "sanity"

#TRAINING_ROOT: "gs://dodam_exp/coarsener"

// #FULL_STATE_CKPT wile load the WHOLE TRAINING STATE.
// This will resume training from the checkpoint AND DISREGARD OTHER
// PARAMETERS IN THIS FILE, such as new learning rates etc.
//#FULL_STATE_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"

#FULL_STATE_CKPT: null

//#ENCODER_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
//#DECODER_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#ENCODER_CKPT: null
#DECODER_CKPT: null

#ENC_CV: "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/rigid"

#FIELD_CV: "gs://zetta-lee-fly-cns-001_dodam/madde/fafb_v15_field"

///////////////////////////////////////////////////////////////////
//////////////////////// Training Spec ////////////////////////////
///////////////////////////////////////////////////////////////////

"@type":   "lightning_train"
ckpt_path: #FULL_STATE_CKPT
regime: {
	"@type": "EncodingCoarsenerHighRes"
	lr:      4e-4
	encoder: #ENCODER_ARCH
	decoder: #DECODER_ARCH
	apply_counts: [1]
	residual_range: [0.25, 5.0]
	residual_weight: 1.0
	field_scale: [1.0, 1.0]
	field_weight:      5.0
	meanstd_weight:    100.0
	encoder_ckpt_path: #ENCODER_CKPT
	decoder_ckpt_path: #DECODER_CKPT
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
#ENCODER_ARCH: {
	"@type": "torch.nn.Sequential"
	modules: [
		{
			"@type": "ConvBlock"
			num_channels: [1, 16, 32]
			activate_last: true
			kernel_sizes:  3
		},
		{
			"@type":             "torch.nn.AvgPool2d"
			"kernel_size":       2
			"count_include_pad": false
		},
		{
			"@type": "ConvBlock"
			num_channels: [32, 32, 1]
			activate_last: true
			kernel_sizes:  3
		},
	]
}
#DECODER_ARCH: {
	"@type": "torch.nn.Sequential"
	modules: [
		{
			"@type": "ConvBlock"
			num_channels: [1, 32, 32]
			activate_last: true
			kernel_sizes:  3
		},
		{
			"@type":        "torch.nn.Upsample"
			"scale_factor": 2
		},
		{
			"@type": "ConvBlock"
			num_channels: [32, 16, 1]
			activate_last: true
			kernel_sizes:  3
		},
	]
}

///////////////////////////////////////////////////////////////////
///////////////////////// Dataset Spec ////////////////////////////
///////////////////////////////////////////////////////////////////
#dset_settings: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		image: {
			"@type": "LayerDataset"
			layer: {
				"@type": "build_layer_set"
				layers: {
					data_in: {
						"@type": "build_cv_layer"
						path:    #ENC_CV
						cv_kwargs: {cache: true}
						read_postprocs: [
							{
								"@type": "rearrange"
								"@mode": "partial"
								pattern: "1 x y z -> z x y"
							},
						]
					}
				}
			}
			sample_indexer: {
				"@type": "VolumetricStepIndexer"
				desired_resolution: [32, 32, 45]
				resolution: [32, 32, 45]
				chunk_size: [768, 768, 1]
				step_size: [384, 384, 1]
				bcube: {
					"@type":     "BoundingCube"
					start_coord: _
					end_coord:   _
					resolution: [4, 4, 45]
				}
			}
		}
		field: {
			"@type": "LayerDataset"
			layer: {
				"@type": "build_layer_set"
				layers: {
					data_in: {
						"@type": "build_cv_layer"
						path:    #FIELD_CV
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
				"@type": "RandomIndexer"
				inner_indexer: {
					"@type": "VolumetricStepIndexer"
					desired_resolution: [128, 128, 40]
					resolution: [128, 128, 40]
					chunk_size: [768, 768, 1]
					step_size: [64, 64, 1]
					bcube: {
						"@type":     "BoundingCube"
						start_coord: _
						end_coord:   _
						resolution: [4, 4, 40]
					}
				}
			}
		}
	}
}

#train_dset: #dset_settings & {
	datasets: {
		image: {
			sample_indexer: {
				bcube: {
					"@type": "BoundingCube"
					start_coord: [98304, 32768, 2500]
					end_coord: [164340, 65536, 3499]
					resolution: [4, 4, 45]
				}
			}
		}
		field: {
			sample_indexer: {
				inner_indexer: {
					bcube: {
						"@type": "BoundingCube"
						start_coord: [98304, 32768, 2000]
						end_coord: [131072, 65536, 3999]
						resolution: [4, 4, 40]
					}
				}
			}
		}
	}
}

#val_dset: #dset_settings & {
	datasets: {
		image: {
			sample_indexer: {
				bcube: {
					"@type": "BoundingCube"
					start_coord: [98304, 32768, 3499]
					end_coord: [114688, 65536, 3500]
					resolution: [4, 4, 45]
				}
			}
		}
		field: {
			sample_indexer: {
				inner_indexer: {
					bcube: {
						"@type": "BoundingCube"
						start_coord: [98304, 32768, 3999]
						end_coord: [131072, 65536, 4000]
						resolution: [4, 4, 40]
					}
				}
			}
		}
	}
}
train_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  4
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
