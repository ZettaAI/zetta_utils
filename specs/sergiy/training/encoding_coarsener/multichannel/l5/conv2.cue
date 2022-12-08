#EXP_NAME:    "encoding_coarsener_multichannel"
#EXP_VERSION: "simple_recons_m6_downs5_conv2_k3_c3_x0"

#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts/cns_coarsener"



//#ENCODER_CKPT:    "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x3/last.ckpt"
//#DECODER_CKPT:    "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x3/last.ckpt"
#ENCODER_CKPT: null
#DECODER_CKPT: null

#ENC_CV: "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/rigid"

///////////////////////////////////////////////////////////////////
//////////////////////// Training Spec ////////////////////////////
///////////////////////////////////////////////////////////////////

#DOWNSAMPLING: {
	"@type":     "torch.nn.AvgPool2d"
	kernel_size: 2
}
#UPSAMPLING: {
	"@type":      "torch.nn.Upsample"
	mode:         "bilinear"
	scale_factor: 2
}

"@type":   "lightning_train"
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
	apply_counts: [1]
	equivar_angle_range: [1, 359]
	equivar_translate_range: [-2, 2]
	equivar_scale_range: [0.95, 1.05]
	equivar_shear_range: [-5, 5]
	equivar_mse_weight: 0.0
	invar_mse_weight:   0.0
	diffkeep_angle_range: [-5, 5]
	diffkeep_weight: 0.0
	fft_weight:      0.0
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

#LEVELS:               5
#INPUT_CHANNELS:       1
#ENC_CHANNELS:         3
#KERNEL_SIZE:          3
#NUM_CONVS:            2
#HIDDEN_CHANNEL_NUM:   32
#HIDDEN_CONV_CHANNELS: [#HIDDEN_CHANNEL_NUM] * #NUM_CONVS

#MID_BLOCK: {
	"@type":       "ConvBlock"
	num_channels:  [#HIDDEN_CHANNEL_NUM] + #HIDDEN_CONV_CHANNELS
	kernel_sizes:  #KERNEL_SIZE
	activate_last: true
}

#FIRST_ENC_BLOCK: {
	"@type":       "ConvBlock"
	num_channels:  [#INPUT_CHANNELS] + #HIDDEN_CONV_CHANNELS
	kernel_sizes:  #KERNEL_SIZE
	activate_last: true
}
#LAST_ENC_BLOCK: {
	"@type":       "ConvBlock"
	num_channels:  #HIDDEN_CONV_CHANNELS + [#ENC_CHANNELS]
	kernel_sizes:  #KERNEL_SIZE
	activate_last: true
}
#FIRST_DEC_BLOCK: {
	"@type":       "ConvBlock"
	num_channels:  [#ENC_CHANNELS] + #HIDDEN_CONV_CHANNELS
	kernel_sizes:  #KERNEL_SIZE
	activate_last: true
}
#LAST_DEC_BLOCK: {
	"@type":       "ConvBlock"
	num_channels:  #HIDDEN_CONV_CHANNELS + [#INPUT_CHANNELS]
	kernel_sizes:  #KERNEL_SIZE
	activate_last: true
}

///////////////////////////////////////////////////////////////////
////////////////////// Architecture Spec //////////////////////////
///////////////////////////////////////////////////////////////////
#ENCODER_MODEL: {
	"@type": "torch.nn.Sequential"
	modules: [#FIRST_ENC_BLOCK, #DOWNSAMPLING] + [#MID_BLOCK, #DOWNSAMPLING]*(#LEVELS-1) + [#LAST_ENC_BLOCK]
}

#DECODER_MODEL: {
	"@type": "torch.nn.Sequential"
	modules: [#FIRST_DEC_BLOCK, #UPSAMPLING] + [#MID_BLOCK, #UPSAMPLING]*(#LEVELS-1) + [#LAST_DEC_BLOCK]
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
		"@type": "VolumetricStridedIndexer"
		chunk_size: [1024, 1024, 1]
		stride: [512, 512, 1]
		resolution: [256, 256, 45]
		desired_resolution: [256, 256, 45]
		bcube: {
			"@type":     "BoundingCube"
			start_coord: _
			end_coord:   _
			resolution: [4, 4, 45]
		}
	}
}

#train_dset: #dset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [0 * 1024, 0 * 1024, 2900]
			end_coord: [196608, 65536, 3000]
			resolution: [4, 4, 45]
		}
	}
}

#val_dset: #dset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [0 * 1024, 0 * 1024, 3100]
			end_coord: [196608, 65536, 3110]
			resolution: [4, 4, 45]
		}
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
