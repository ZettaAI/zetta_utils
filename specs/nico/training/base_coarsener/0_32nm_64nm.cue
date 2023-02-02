#EXP_NAME:      "base_coarsener"
#TRAINING_ROOT: "gs://zetta-research-nico/training_artifacts"
#POST_WEIGHT:   1.55
#POST_WEIGHT_MULT:   1.05
#ZCONS_WEIGHT:  0.0
#LR:            1e-4
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   0.5
#EXP_VERSION:   "tmp_ft_patch\(#CHUNK_XY)_post\(#POST_WEIGHT)_postmult\(#POST_WEIGHT_MULT)_lr\(#LR)_deep_k\(#K)_clip\(#CLIP)_equi\(#EQUI_WEIGHT)_fmt\(#FIELD_MAGNITUDE_THRESHOLD)_f1f2_apply_1_m3_int8_more_data_4"
#CHUNK_XY:      1024
#FIELD_MAGNITUDE_THRESHOLD: 1.1

#START_EXP_VERSION: "ft_patch1024_post1.25_postmult1.05_lr0.0001_deep_k3_clip0.00000_equi0.5_fmt1.1_f1f2_apply_1234_m3_int8_more_data_2"
#MODEL_CKPT:        "gs://zetta-research-nico/training_artifacts/base_coarsener/\(#START_EXP_VERSION)/last.ckpt"


"@type":        "mazepa.execute_on_gcp_with_sqs"
worker_image:   "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_neighbors"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     1
local_test: true

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "BaseCoarsenerRegime"
		field_magn_thr:         #FIELD_MAGNITUDE_THRESHOLD
		max_displacement_px:    16.0
		val_log_row_interval:   4
		train_log_row_interval: 200
		lr:                     #LR
		zero_value:             0.0
		equivar_weight:         #EQUI_WEIGHT
		post_weight:            #POST_WEIGHT
		post_weight_multiplier: #POST_WEIGHT_MULT
		zero_conserve_weight:   #ZCONS_WEIGHT
		apply_counts: [1]
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
		val_check_interval:      600
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
                        path:    "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns"
                        cv_kwargs: {
                            // cache: "/home/nkemnitz/.cloudvolume/memcache"
                        }
                        read_procs: #IMG_PROCS
                    }
                    tgt: {
                        "@type": "build_cv_layer"
                        path:    "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns"
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
            sample_indexer: {
                "@type": "VolumetricStridedIndexer"
                resolution: [32, 32, 45]
                desired_resolution: [32, 32, 45]
                chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
                stride: [#CHUNK_XY, #CHUNK_XY, 1]
                bbox: {
                    "@type":       "BBox3D.from_coords"
                    "start_coord": _
                    "end_coord":   _
                    resolution:    _
                }
            }
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
		"start_coord": [0, 0, 2018]
	}
	"stride": [64, 64, 1]
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
	// datasets: src: datasets: zfish: sample_indexer: bbox: {
	// 	start_coord: [256, 256, 0]
	// 	end_coord: [1024, 1024, 40]
	// 	resolution: [256, 256, 30]
	// }
	// datasets: src: datasets: cns: sample_indexer: bbox: {
	datasets: images: sample_indexer: bbox: {
		start_coord: [1280, 0, 2700]
		end_coord: [2816, 3584, 3100]
		resolution: [256, 256, 45]
	}
}

#val_dset: #dset_settings & {
	datasets: field: sample_indexer: {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
	_img_procs: #IMG_PROCS
	// datasets: src: datasets: zfish: sample_indexer: bbox: {
	// 	start_coord: [512, 512, 40]
	// 	end_coord: [1024, 1024, 41]
	// 	resolution: [256, 256, 30]
	// }
	// datasets: src: datasets: cns: sample_indexer: bbox: {
	datasets: images: sample_indexer: bbox: {
		start_coord: [2048, 256, 3180]
		end_coord: [2512, 1024, 3182]
		resolution: [256, 256, 45]
	}
}