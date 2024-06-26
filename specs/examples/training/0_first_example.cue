#EXP_NAME:      "mimic_encodings"
#EXP_VERSION:   "tile_aug_x5"
#TRAINING_ROOT: "gs://tmp_2w/nkem/training_artifacts"

//#MODEL_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"
#MODEL_CKPT: null // Set to a path to only load the net weights

#IMG_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
#ENC_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

///////////////////////////////////////////////////////////////////
//////////////////////// Training Spec ////////////////////////////
///////////////////////////////////////////////////////////////////

"@type": "lightning_train"
image: "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:sergiy_x0013"
resource_limits: {
    memory:           "18560Mi"
    "nvidia.com/gpu": "1"
}
num_nodes: 1
local_run:      true
regime: {
	"@type": "NaiveSupervisedRegime"
	train_log_row_interval: 100
	lr:      4e-4
	model: {
		"@type": "load_weights_file"
		model: {
			"@type": "ConvBlock"
			num_channels: [1, 32, 32, 32, 32, 1]
			kernel_sizes: [5, 5]
			skips: {"0": 3}
		}
		ckpt_path: #MODEL_CKPT
		component_names: [
			"model",
		]
	}
}

trainer: {
	"@type":            "ZettaDefaultTrainer"
	accelerator:        "gpu"
	devices:            1
	max_epochs:         1
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  10
	val_check_interval: 100
	checkpointing_kwargs: {
		update_every_n_secs: 60 * 3
		backup_every_n_secs: 60 * 20
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
				read_procs: [
					{
						"@type": "divide"
						"@mode": "partial"
						value:   255.0
					},
					{
						"@type": "square_tile_pattern_aug"
						"@mode": "partial"
						prob:    1.0
						tile_size: {
							"@type": "uniform_distr"
							low:     64
							high:    1024
						}
						tile_stride: {
							"@type": "uniform_distr"
							low:     64
							high:    1024
						}
						max_brightness_change: {
							"@type": "uniform_distr"
							low:     0.2
							high:    0.4
						}
						rotation_degree: {
							"@type": "uniform_distr"
							low:     0
							high:    45
						}
						preserve_data_val: 0.0
						repeats:           3
						device:            "cpu"
					},
					{
						"@type": "add"
						"@mode": "partial"
						value:   -0.5
					},
					{
						"@type":    "clamp_values_aug"
						"@mode":    "partial"
						prob:       1.0
						low_distr:  -0.5
						high_distr: 0.5
					},
					{
						"@type": "rearrange"
						"@mode": "partial"
						pattern: "c x y 1 -> c x y"
					},
				]
			}
			target: {
				"@type": "build_cv_layer"
				path:    #ENC_CV
				//cv_kwargs: {cache: true}
				read_procs: [
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
		resolution: [64, 64, 40]
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
			start_coord: [80000, 30000, 2000]
			end_coord: [230000, 80000, 2099]
			resolution: [4, 4, 40]
		}
	}
}

#val_dset: #dset_settings & {
	sample_indexer: {
		bbox: {
			"@type": "BBox3D.from_coords"
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
