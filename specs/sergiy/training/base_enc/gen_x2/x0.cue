#EXP_NAME:      "debug"
#TRAINING_ROOT: "gs://tmp_2w"
#POST_WEIGHT:   1.55
#ZCONS_WEIGHT:  0.0
#LR:            2e-4
#CLIP:          0e-5
#K:             3
#EQUI_WEIGHT:   0.5
#CHUNK_XY:      1024
#GAMMA_LOW:     0.25
#GAMMA_HIGH:    4.0
#GAMMA_PROB:    1.0
#TILE_LOW:      0.1
#TILE_HIGH:     0.4

#EXP_VERSION: "debug_old_image_new_zutils"

#START_EXP_VERSION: "ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17"
#MODEL_CKPT:        null// "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#START_EXP_VERSION)/last.ckpt"

#FIELD_CV: "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:old_img_new_zutils_local"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas: 1
local_test:      true

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "BaseEncoderRegime"
		field_magn_thr:         0.8
		val_log_row_interval:   1
		train_log_row_interval: 150
		lr:                     #LR
		equivar_weight:         #EQUI_WEIGHT
		post_weight:            #POST_WEIGHT
		zero_conserve_weight:   #ZCONS_WEIGHT
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
					},
					{
						"@type": "torch.nn.Tanh"
					},
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
		max_epochs:              100
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      100
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
#VAL_IMG_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   255.0
	},
]
#TRAIN_IMG_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
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
			low:     #TILE_LOW
			high:    #TILE_HIGH
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
]

#ORIGINAL_ZFISH: {
	"@type":    "LayerDataset"
	_img_procs: _
	layer: {
		"@type": "build_layer_set"
		layers: {
			src: {
				"@type":    "build_cv_layer"
				path:       "gs://sergiy_exp/pairs_dsets/zfish_x0/src"
				read_procs: _img_procs
			}
			tgt: {
				"@type":    "build_cv_layer"
				path:       "gs://sergiy_exp/pairs_dsets/zfish_x0/dst"
				read_procs: _img_procs
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricNGLIndexer"
		resolution: [32, 32, 30]
		desired_resolution: [32, 32, 30]
		chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
		path: _
	}
}

#CUTOUT_X0: {
	"@type":    "LayerDataset"
	_img_procs: _
	layer: {
		"@type": "build_layer_set"
		layers: {
			src: {
				"@type":    "build_cv_layer"
				path:       "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
				read_procs: _img_procs
			}
			tgt: {
				"@type":    "build_cv_layer"
				path:       "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/imgs_warped/-1"
				read_procs: _img_procs
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStridedIndexer"
		resolution: [32, 32, 30]
		desired_resolution: [32, 32, 30]
		stride: [#CHUNK_XY, #CHUNK_XY, 1]
		chunk_size: [#CHUNK_XY, #CHUNK_XY, 1]
		bbox: {
			"@type":     "BBox3D.from_coords"
			end_coord:   _
			resolution:  _
			start_coord: _
		}

	}
}
#dset_settings: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		field: #FIELD_DSET
		images: {
			"@type":  "JointDataset"
			mode:     "vertical"
			datasets: _
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
	"sample_indexer": {
		"@type":       "RandomIndexer"
		inner_indexer: #field_indexer
	}
}

#field_indexer: {
	"@type": "VolumetricStridedIndexer"
	"bbox": {
		"@type": "BBox3D.from_coords"
		"end_coord": [
			2048,
			2048,
			2040,
		]
		"resolution": [
			4,
			4,
			45,
		]
		"start_coord": [
			0,
			0,
			2030,
		]
	}
	"stride": [
		64,
		64,
		1,
	]
	"chunk_size": [
		#CHUNK_XY,
		#CHUNK_XY,
		1,
	]
	"resolution": [
		4,
		4,
		45,
	]
	"desired_resolution": [
		4,
		4,
		45,
	]
}

#train_dset: #dset_settings & {
	datasets: images: datasets: {
		original: #ORIGINAL_ZFISH & {
			sample_indexer: path: "sergiy/base_enc_train_x0"
			_img_procs: #TRAIN_IMG_PROCS
		}
		cutout_x0_part0: #CUTOUT_X0 & {
			sample_indexer: bbox: {
				resolution: [4, 4, 30]
				start_coord: [1024 * 25, 1024 * 30, 130]
				end_coord: [1024 * 75, 1024 * 100, 155]
			}
			_img_procs: #TRAIN_IMG_PROCS
		}
		cutout_x0_part1: #CUTOUT_X0 & {
			_img_procs: #TRAIN_IMG_PROCS
			sample_indexer: bbox: {
				resolution: [4, 4, 30]
				start_coord: [1024 * 25, 1024 * 30, 170]
				end_coord: [1024 * 75, 1024 * 100, 200]
			}
		}

	}
}

#val_dset: #dset_settings & {

	datasets: images: datasets: {
		original: #ORIGINAL_ZFISH & {
			sample_indexer: path: "sergiy/base_enc_val_bright_x0"
			_img_procs: #VAL_IMG_PROCS
		}
	}
}
