#EXP_NAME:      "base_coarsener_simplified"
#TRAINING_ROOT: "gs://zetta-research-nico/training_artifacts"
#POST_WEIGHT:   1.0
#LR:            1e-4
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   0.5
#EXP_VERSION:   "tmp_ft_patch\(#CHUNK_XY)_post\(#POST_WEIGHT)_lr\(#LR)_deep_k\(#K)_clip\(#CLIP)_equi\(#EQUI_WEIGHT)_fmt\(#FIELD_MAGNITUDE_THRESHOLD)_f1f2_m4"
#CHUNK_XY:      1024
#FIELD_MAGNITUDE_THRESHOLD: 1.41

#START_EXP_VERSION: "tmp_ft_patch1024_post1.0_lr0.0001_deep_k3_clip0.00000_equi0.5_fmt1.1_f1f2_m4"
#MODEL_CKPT:        "gs://zetta-research-nico/training_artifacts/base_coarsener_simplified/\(#START_EXP_VERSION)/last.ckpt"


"@type":        "mazepa.execute_on_gcp_with_sqs"
worker_image:   "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_unet_pow"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     1
local_test: false

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "BaseCoarsenerSimplifiedRegime"
		field_magn_thr:         #FIELD_MAGNITUDE_THRESHOLD
		post_weight:            #POST_WEIGHT
		max_displacement_px:    16.0
		val_log_row_interval:   8
		train_log_row_interval: 500
		lr:                     #LR
		zero_value:             0.0
		equivar_weight:         #EQUI_WEIGHT
		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "UNet"
						list_num_channels: [
							[1, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],

							[32, 32, 32],

							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 32],
							[32, 32, 1],
						]
						kernel_sizes: [#K, #K]
						padding_modes: "reflect"
					},
					{
						"@type":     "torch.nn.AvgPool2d",
						kernel_size: 2
					},
					{
						"@type": "torch.nn.Tanh"
					}
				]
			}
			ckpt_path: #MODEL_CKPT
			component_names: [
				"model",
			]
		}
	}
	trainer: {
		"@type":                 "ZettaDefaultTrainer"
		accelerator:             "gpu"
		devices:                 1
		max_epochs:              1
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      500
		gradient_clip_algorithm: "norm"
		gradient_clip_val:       #CLIP
		checkpointing_kwargs: {
			update_every_n_secs: 60
			backup_every_n_secs: 900
		}
	}

	train_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     true
		num_workers: 4
		dataset:     #train_dset
	}
	val_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  1
		shuffle:     false
		num_workers: 4
		dataset:     #val_dset
	}
}
#IMG_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   127.0
	}
]

#dset_settings: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		field: #FIELD_DSET
		images: {
            "@type": "LayerDataset"
            layer: {
                "@type": "build_layer_set"
                layers: {
                    src: {
                        "@type": "build_cv_layer"
                        path:    "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns_64nm_unet_pow_post1.1"
                        cv_kwargs: {
                            // cache: "/home/nkemnitz/.cloudvolume/memcache"
                        }
                        read_procs: #IMG_PROCS
                    }
                    tgt: {
                        "@type": "build_cv_layer"
                        path:    "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns_64nm_unet_pow_post1.1"
                        cv_kwargs: {
                            // cache: "/home/nkemnitz/.cloudvolume/memcache"
                        }
                        index_procs: [
                            {
                                "@type": "VolumetricIndexTranslator"
                                offset: [0, 0, -1]
                                resolution: [4, 4, 45]
                            },
                        ]
                        read_procs: #IMG_PROCS
                    }
                }
            }
            "sample_indexer": _
		}
	}
}

#FIELD_DSET: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_cv_layer"
		path:    "gs://zetta-research-nico/perlin_noise_fields/1px"
		read_procs: [
			{
				"@type":   "rearrange"
				"@mode":   "partial"
				"pattern": "c x y 1 -> c x y"
			},
		]
	}
	"sample_indexer": _
}

#field_indexer: {
	"@type": "VolumetricStridedIndexer"
	"bbox": {
		"@type": "BBox3D.from_coords"
		"end_coord": [2048, 2048, 2040]
		"resolution": [4, 4, 45]
		"start_coord": [0, 0, 0]
	}
	"stride": [512, 512, 1]
	"chunk_size": [#CHUNK_XY, #CHUNK_XY, 1]
	"resolution": [4, 4, 45]
	"desired_resolution": [4, 4, 45]
}

#train_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: #IMG_PROCS
	datasets: images: sample_indexer: {
		"@type": "RandomIndexer"
		inner_indexer: {
			"@type": "VolumetricStridedIndexer"
			resolution: [64, 64, 45]
			desired_resolution: [64, 64, 45]
			chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
			stride: [#CHUNK_XY, #CHUNK_XY, 1]
			bbox: {
				"@type":       "BBox3D.from_coords"
				start_coord: [1280, 0, 2701]
				end_coord: [2816, 3584, 3180]
				resolution: [256, 256, 45]
			}
		}
	}
}

#val_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: #IMG_PROCS
	datasets: images: sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [64, 64, 45]
		desired_resolution: [64, 64, 45]
		chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
		stride: [#CHUNK_XY, #CHUNK_XY, 1]
		bbox: {
			"@type":       "BBox3D.from_coords"
			start_coord: [2048, 256, 3180]
			end_coord: [2512, 1024, 3188]
			resolution: [256, 256, 45]
		}
	}
}