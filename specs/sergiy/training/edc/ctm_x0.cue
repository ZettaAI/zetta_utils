#EXP_NAME:      "ctm"
#TRAINING_ROOT: "gs://sergiy_exp/training_artifacts"
#LR:            16e-5
#CLIP:          0e-5
#MAXBATCH: 40
#EXP_VERSION:   "lr\(#LR)_cl\(#CLIP)_enc\(#SEG_ENC_DIM)_rescale_maxbatch\(#MAXBATCH)_shift\(#SHIFT_MAGN)_noval_x13"
#K: 3
#FOV_Z: 5
#SEG_ENC_DIM:0
#SHIFT_MAGN:  5
#CHUNK_SIZE:    [90, 90, 5 + #FOV_Z * 2]

#MODEL_CKPT: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"

"@type": "lightning_train"
image: "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:sergiy_x0013"
resource_limits: {
    memory:           "18560Mi"
    "nvidia.com/gpu": "1"
}
num_nodes: 1
local_run:      false

regime: #REGIME
trainer: {
    "@type":                 "ZettaDefaultTrainer"
    accelerator:             "gpu"
    devices:                 1
    max_epochs:              1000
    default_root_dir:        #TRAINING_ROOT
    experiment_name:         #EXP_NAME
    experiment_version:      #EXP_VERSION
    log_every_n_steps:       50
    val_check_interval:      500
    gradient_clip_algorithm: "norm"
    gradient_clip_val:       #CLIP
    checkpointing_kwargs: {
        update_every_n_secs: 300
        backup_every_n_secs: 60 * 20
    }
}

train_dataloader: {
    "@type":     "TorchDataLoader"
    batch_size:  1
    shuffle:     true
    num_workers: 4
    dataset:     #TRAIN_DSET
}
val_dataloader: {
    "@type":     "TorchDataLoader"
    batch_size:  1
    shuffle:     false
    num_workers: 2
    dataset:     #VAL_DSET
}
#ENCODER: {
    "@type": "torch.nn.Sequential"
    modules: [
        {
            "@type": "ConvBlock"
            num_channels: [1 + #SEG_ENC_DIM, 32, 32]
            conv: {
                "@type": "torch.nn.Conv3d"
                "@mode": "partial"
            }
            activate_last: true
            kernel_sizes: [#K, #K, #K]
        },
        {
            "@type":     "torch.nn.AvgPool3d"
            kernel_size: [2, 2, 1]
        },
        {
            "@type": "ConvBlock"
            num_channels: [32, 32, 32]
            conv: {
                "@type": "torch.nn.Conv3d"
                "@mode": "partial"
            }
            activate_last: true
            kernel_sizes: [#K, #K, #K]
        },
        {
            "@type":     "torch.nn.AvgPool3d"
            kernel_size: [2, 2, 1]
        },
        {
            "@type": "ConvBlock"
            num_channels: [32, 16, 8]
            conv: {
                "@type": "torch.nn.Conv3d"
                "@mode": "partial"
            }
            activate_last: true
            kernel_sizes: [#K, #K, #K]
        },
    ],
}

#AGGREGATOR: {
    "@type": "torch.nn.Sequential"
    modules: [
        {
            "@type": "torch.nn.Linear",
            in_features: 12 * 12 * #FOV_Z * 8 * 2
            out_features: 64
        },
        {
            "@type": "torch.nn.LeakyReLU"
        },
        {
            "@type": "torch.nn.Linear",
            in_features: 64
            out_features: 1
        },
    ]
}

#REGIME: {
    "@type":                "CTMRegimev0"
    val_log_row_interval:   1
    train_log_row_interval: 250
    lr:                     #LR
    max_batch_num: #MAXBATCH
    gap_z: 5
    fov_z: 5
    fov_xy: 48
    min_crossection: 6
    min_seg_size: 20
    seg_enc_dim: #SEG_ENC_DIM
    x_shift: {
        "@type": "uniform_distr"
        low:     -#SHIFT_MAGN
        high:    #SHIFT_MAGN
    }
    y_shift: {
        "@type": "uniform_distr"
        low:     -#SHIFT_MAGN
        high:    #SHIFT_MAGN
    }

    encoder: {
        "@type": "load_weights_file"
        model: #ENCODER
        ckpt_path: #MODEL_CKPT
        component_names: [
            "encoder",
        ]
    }
    aggregator: {
        "@type": "load_weights_file"
        model: #AGGREGATOR
        ckpt_path: #MODEL_CKPT
        component_names: [
            "aggregator",
        ]
    }
}


#dset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_layer_set"
		layers: {
			data_in: {
				"@type": "build_cv_layer"
				path:    "gs://zheng_mouse_hippocampus_scratch_30/make_cv_happy/seg/v0.1-16nm-updown_24-24-45_20240520021649"
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [24, 24, 45]
		chunk_size: #CHUNK_SIZE
		stride:     _
		bbox: {
			"@type":     "BBox3D.from_coords"
			start_coord: _
			end_coord:   _
			resolution: [
				12,
				12,
				45,
			]
		}
	}
}

#TRAIN_DSET: #dset_settings & {
	sample_indexer: stride: [64, 64, 1]
	//sample_indexer: stride: #CHUNK_SIZE
	sample_indexer: bbox: {
		start_coord: [1024 * 65, 1024 * 82, 800]
		end_coord: [1024 * 94, 1024 * 105, 900]
	}
}

#VAL_DSET: #dset_settings & {
	sample_indexer: stride: #CHUNK_SIZE
	sample_indexer: bbox: {
		start_coord: [1024 * 70, 1024 * 80, 1000]
		end_coord: [1024 * 72, 1024 * 82, 1015]
	}
}
