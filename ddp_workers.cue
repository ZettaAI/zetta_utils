#EXP_NAME:    "encoding_coarsener"
#EXP_VERSION: "ddp_multinode_x0"

#TRAINING_ROOT: "gs://tmp_2w/akhilesh/training_artifacts"

//#ENCODER_CKPT:    "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x3/last.ckpt"
//#DECODER_CKPT:    "\(#TRAINING_ROOT)/\(#EXP_NAME)/inver_diffkeep_apply2x_x3/last.ckpt"
#ENCODER_CKPT: null
#DECODER_CKPT: null


#ENC_CV: "gs://fafb_v15_aligned/v0/img/img"

"@type":      "multinode_train_launch"
num_nodes: 4
rdzv_endpoint: "master:29400"

///////////////////////////////////////////////////////////////////
//////////////////////// Training Spec ////////////////////////////
///////////////////////////////////////////////////////////////////

regime: {
	"@type": "EncodingCoarsenerRegime"
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
	accelerator:        "cuda"
	devices:            1
	num_nodes:          4
	max_epochs:         100
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  75
	val_check_interval: 75
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
	num_workers: 0
	dataset:     #train_dset
}
val_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  1
	shuffle:     false
	num_workers: 0
	dataset:     #val_dset
}

///////////////////////////////////////////////////////////////////
////////////////////// Architecture Spec //////////////////////////
///////////////////////////////////////////////////////////////////
#ENCODER_MODEL: {
	"@type": "torch.nn.Sequential"
	modules: [
		{
			"@type": "ConvBlock"
			num_channels: [1, 32, 32, 32]
			kernel_sizes: [3, 3]
		},
		{
			"@type":     "torch.nn.AvgPool2d"
			kernel_size: 2
		},
		{
			"@type": "ConvBlock"
			num_channels: [32, 32, 32, 1]
			kernel_sizes: [3, 3]
		},
	]
}

#DECODER_MODEL: {
	"@type": "torch.nn.Sequential"
	modules: [
		{
			"@type": "ConvBlock"
			num_channels: [1, 32, 32, 32]
			kernel_sizes: [3, 3]
		},
		{
			"@type":      "torch.nn.Upsample"
			scale_factor: 2
			mode:         "bilinear"
		},
		{
			"@type": "ConvBlock"
			num_channels: [32, 32, 32, 1]
			kernel_sizes: [3, 3]
		},
	]
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
				"@type": "build_ts_layer"
				path:    #ENC_CV
				//cv_kwargs: {cache: true}
				read_procs: [
					{
						"@type": "rearrange"
						"@mode": "partial"
						pattern: "c x y 1 -> c x y"
					},
					{
						"@type": "to_float32"
						"@mode": "partial"
					}
				]
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		chunk_size: [1024, 1024, 1]
		stride: [512, 512, 1]
		resolution: [128, 128, 40]
		bbox: {
			"@type":     "BBox3D.from_coords"
			start_coord: _
			end_coord:   _
			resolution: [4, 4, 40]
		}
	}
}

#train_dset: #dset_settings & {
	sample_indexer: {
		bbox: {
			"@type": "BBox3D.from_coords"
			start_coord: [64 * 1024, 32 * 1024, 2000]
			end_coord: [128 * 1024, 64 * 1024, 2099]
			resolution: [4, 4, 40]
		}
	}
}

#val_dset: #dset_settings & {
	sample_indexer: {
		bbox: {
			"@type": "BBox3D.from_coords"
			start_coord: [80 * 1024, 30 * 1024, 2099]
			end_coord: [230000, 80000, 2100]
			resolution: [4, 4, 40]
		}
	}
}
