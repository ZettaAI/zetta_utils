#EXP_NAME:    "misalignmentdetector"
#EXP_VERSION: "sanity2"

#TRAINING_ROOT: "gs://dodam_exp/training_artifacts"

// This will resume training from the checkpoint AND DISREGARD OTHER
// PARAMETERS IN THIS FILE, such as new learning rates etc.


//#ENC_CV: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v2_restitch/encodings_decay150_tile_aug"
#ENC_CV:   "gs://fafb_v15_aligned/v0/img/img_norm"
#FIELD_CV: "gs://zetta-lee-fly-cns-001_dodam/madde/fafb_v15_field"

#MAX_DISP: 8.0

///////////////////////////////////////////////////////////////////
//////////////////////// Training Spec ////////////////////////////
///////////////////////////////////////////////////////////////////

"@type":   "lightning_train"
regime: {
	"@type":          "MisalignmentDetector"
	detector:         #DETECTOR_ARCH
	lr:               4e-4
	max_disp:         #MAX_DISP
	downsample_power: 0
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
#DETECTOR_ARCH: {
	"@type": "torch.nn.Sequential"
	modules: [
		{
			"@type": "UNet"
			list_num_channels: [[2, 8, 8], [16, 16, 16], [32, 32, 32], [16, 16, 16], [8, 8, 1]]
			downsample: {
				"@type":       "torch.nn.Conv2d"
				"@mode":       "partial"
				"kernel_size": 3
				"stride":      2
				"device":      "cuda"
				"padding": {1, 1}
			}
			upsample: {
				"@type":       "torch.nn.ConvTranspose2d"
				"@mode":       "partial"
				"kernel_size": 2
				"stride":      2
				"device":      "cuda"
			}
			normalization: {
				"@type":  "torch.nn.BatchNorm2d"
				"@mode":  "partial"
				"device": "cuda"
			}
			kernel_sizes:   3
			strides:        1
			paddings:       1
			activate_last:  false
			normalize_last: false
		},
		{
			"@type": "torch.nn.Sigmoid"
		},
		{
			"@type": "RescaleValues"
			in_range: [0, 1]
			out_range: [0, #MAX_DISP]
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
						//cv_kwargs: {cache: true}
						read_procs: [
							{
								"@type":   "rearrange"
								"@mode":   "partial"
								"pattern": "1 x y z -> z x y"
							},
						]
					}
				}
			}
			sample_indexer: {
				"@type": "VolumetricStridedIndexer"
				desired_resolution: [64, 64, 40]
				index_resolution: [64, 64, 40]
				resolution: [64, 64, 40]
				chunk_size: [512, 512, 2]
				stride: [512, 512, 1]
				bbox: {
					"@type":     "BBox3D.from_coords"
					start_coord: _
					end_coord:   _
					resolution: [4, 4, 40]
				}
			}
		}
		field0: {
			"@type": "LayerDataset"
			layer: {
				"@type": "build_layer_set"
				layers: {
					data_in: {
						"@type": "build_cv_layer"
						path:    #FIELD_CV
						//cv_kwargs: {cache: true}
						read_procs: [
							{
								"@type":   "rearrange"
								"@mode":   "partial"
								"pattern": "c x y 1 -> c x y"
							},
						]
					}
				}
			}
			sample_indexer: {
				"@type": "RandomIndexer"
				inner_indexer: {
					"@type": "VolumetricStridedIndexer"
					desired_resolution: [128, 128, 40]
					index_resolution: [128, 128, 40]
					resolution: [128, 128, 40]
					chunk_size: [512, 512, 1]
					stride: [512, 512, 1]
					bbox: {
						"@type":     "BBox3D.from_coords"
						start_coord: _
						end_coord:   _
						resolution: [4, 4, 40]
					}
				}
			}
		}
		field1: {
			"@type": "LayerDataset"
			layer: {
				"@type": "build_layer_set"
				layers: {
					data_in: {
						"@type": "build_cv_layer"
						path:    #FIELD_CV
						//cv_kwargs: {cache: true}
						read_procs: [
							{
								"@type":   "rearrange"
								"@mode":   "partial"
								"pattern": "c x y 1 -> c x y"
							},
						]
					}
				}
			}
			sample_indexer: {
				"@type": "RandomIndexer"
				inner_indexer: {
					"@type": "VolumetricStridedIndexer"
					desired_resolution: [128, 128, 40]
					index_resolution: [128, 128, 40]
					resolution: [128, 128, 40]
					chunk_size: [512, 512, 1]
					stride: [512, 512, 1]
					bbox: {
						"@type":     "BBox3D.from_coords"
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
				bbox: {
					"@type": "BBox3D.from_coords"
					start_coord: [98304, 32768, 2000]
					end_coord: [131072, 65536, 2998]
					resolution: [4, 4, 40]
				}
			}
		}
		field0: {
			sample_indexer: {
				inner_indexer: {
					bbox: {
						"@type": "BBox3D.from_coords"
						start_coord: [98304, 32768, 2000]
						end_coord: [131072, 65536, 3999]
						resolution: [4, 4, 40]
					}
				}
			}
		}
		field1: {
			sample_indexer: {
				inner_indexer: {
					bbox: {
						"@type": "BBox3D.from_coords"
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
				bbox: {
					"@type": "BBox3D.from_coords"
					start_coord: [98304, 32768, 2000]
					end_coord: [131072, 65536, 2998]
					resolution: [4, 4, 40]
				}
			}
		}
		field0: {
			sample_indexer: {
				inner_indexer: {
					bbox: {
						"@type": "BBox3D.from_coords"
						start_coord: [98304, 32768, 3998]
						end_coord: [131072, 65536, 3999]
						resolution: [4, 4, 40]
					}
				}
			}
		}
		field1: {
			sample_indexer: {
				inner_indexer: {
					bbox: {
						"@type": "BBox3D.from_coords"
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
