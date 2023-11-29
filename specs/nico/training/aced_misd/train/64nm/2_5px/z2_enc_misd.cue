import "strings"
import "strconv"
import "list"

#EXP_NAME:       "aced_misd_general"
#TRAINING_ROOT:  "gs://zetta-research-nico/training_artifacts"
#LR:             2e-4
#K:              3
#CHUNK_XY:       1024
#FM:             32

#FIELD_MAGN_THR: 2.5
#Z_OFFSETS:      [2]
#DS_FACTOR:      2


#EXP_VERSION: "1.0.0_dsfactor\(#DS_FACTOR)_thr\(#FIELD_MAGN_THR)_lr\(#LR)_z" + strings.Join([for z in #Z_OFFSETS {strconv.FormatInt(z, 10)}], "_")
#MODEL_CKPT: null  // "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr5.0_lr0.00005_z1z2_400-500_2910-2920_more_aligned_unet5_32/last.ckpt"

#BASE_PATH: "gs://zetta-research-nico/encoder/"
#SRC_ENC_PATH: #BASE_PATH + "misd/enc/" // + k + ["/good_alignment"|"/bad_alignment"] + "/z\(_z_offset)"
#TGT_ENC_PATH: #BASE_PATH + "pairwise_aligned/" // + k + "/tgt_enc_2023"
#DISP_PATH: #BASE_PATH + "misd/misalignment_fields/" // + k + "/displacements/z\(_z_offset)"

#VAL_DATASETS: {
	"microns_basil": {"resolution": [32, 32, 40], "num_samples": 2591},
}

#TRAIN_DATASETS: {
	"microns_pinky": {"resolution": [32, 32, 40], "num_samples": 5019},
	// "microns_basil": {"resolution": [32, 32, 40], "num_samples": 2591},
	"kim_n2da": {"resolution": [32, 32, 50], "num_samples": 446},
	"kim_pfc2022": {"resolution": [32, 32, 40], "num_samples": 3699},
	"kronauer_cra9": {"resolution": [32, 32, 42], "num_samples": 740},
	"kubota_001": {"resolution": [40, 40, 40], "num_samples": 4744},
	"lee_ppc": {"resolution": [32, 32, 40], "num_samples": 7219},
	"prieto_godino_larva": {"resolution": [32, 32, 32], "num_samples": 4584},
	"janelia_hemibrain": {"resolution": [32, 32, 32], "num_samples": 5304},
	"nguyen_thomas_2022": {"resolution": [32, 32, 40], "num_samples": 1847},
	"mulcahy_2022_16h": {"resolution": [32, 32, 30], "num_samples": 3379},
	"wildenberg_2021_vta_dat12a": {"resolution": [32, 32, 40], "num_samples": 1704},
	"bumbarber_2013": {"resolution": [31.2, 31.2, 50.0], "num_samples": 7325},
	"wilson_2019_p3": {"resolution": [32, 32, 30], "num_samples": 2092},
	"ishibashi_2021_em1": {"resolution": [32, 32, 32], "num_samples": 141},
	"ishibashi_2021_em2": {"resolution": [32, 32, 32], "num_samples": 166},
	"templier_2019_wafer1": {"resolution": [32, 32, 50], "num_samples": 5401},
	"templier_2019_wafer3": {"resolution": [32, 32, 50], "num_samples": 3577},
	"lichtman_octopus2022": {"resolution": [32, 32, 30], "num_samples": 5673},
}

#UNET_DOWNSAMPLE: {
	"@type": "torch.nn.MaxPool2d"
	"@mode": "partial"
	kernel_size: 2
}

#UNET_UPSAMPLE: {
	{
		"@type": "UpConv"
		"@mode": "partial"
		kernel_size: #K
		upsampler: {
			"@type": "torch.nn.Upsample"
			"@mode": "partial"
			scale_factor: 2
			mode: "nearest"
			align_corners: null
		},
		conv: {
			"@type": "torch.nn.Conv2d"
			"@mode": "partial"
			padding: 1
		}
	}
}

#TARGET: {
	"@type": "lightning_train"
	regime: {
		"@type":                "MisalignmentDetectorAcedRegime"
		output_mode:            "binary"
		encoder_path:           null
		max_src_displacement_px: {
			"@type": "uniform_distr"
			low:     0.0
			high:    0.0
		}
		equivar_rot_deg_distr: {
			"@type": "uniform_distr"
			low:     0.0
			high:    0.0
		}
		equivar_trans_px_distr: {
			"@type": "uniform_distr"
			low:     0.0
			high:    0.0
		}

		field_magn_thr:         #FIELD_MAGN_THR
		val_log_row_interval:   4
		train_log_row_interval: 200
		lr:                     #LR
		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "ConvBlock",
						"@version": "0.0.2"
						num_channels: [2, #FM],
						kernel_sizes: [5, 5],
						activate_last: true,
					},
					{
						"@type": "UNet"
						"@version": "0.0.2"
						list_num_channels: [
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],

							[#FM, #FM, #FM],

							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
							[#FM, #FM, #FM],
						]
						downsample: #UNET_DOWNSAMPLE
						upsample: #UNET_UPSAMPLE
						activate_last: true
						kernel_sizes: [#K, #K]
						padding_modes: "zeros"
						unet_skip_mode: "sum"
						skips: {"0": 2}
					},
					{
						"@type": "torch.nn.Conv2d"
						in_channels: #FM
						out_channels: 1
						kernel_size: 1
					},
					// {
					// 	"@type": "torch.nn.Sigmoid"  # Regime applies binary_cross_entropy_with_logits
					// }
				]
			},
			ckpt_path: #MODEL_CKPT
			component_names: [
				"model",
			]
		}
	}
	trainer: {
		"@type":                 "ZettaDefaultTrainer"
		accelerator:             "gpu"
		precision:               "16-mixed",
		strategy:                "auto",
		devices:                 1
		max_epochs:              100
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		val_check_interval:      250
		num_sanity_val_steps:    -1
		reload_dataloaders_every_n_epochs: 1,
		checkpointing_kwargs: {
			update_every_n_secs: 1700
			backup_every_n_secs: 3700
		}
	}

	train_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  8
		//shuffle:     true
		sampler: {
			"@type": "SamplerWrapper",
			sampler: {
				"@type": "TorchRandomSampler"
				data_source: {
					"@type": "torch.arange"
					"end": list.Sum([for dataset in #TRAIN_DATASETS {dataset.num_samples}])
				},
				replacement: false,
				num_samples: 2000,
			},
		},
		num_workers: 8
		dataset:     #TRAINING
		pin_memory:  true
	}
	val_dataloader: {
		"@type":     "TorchDataLoader"
		batch_size:  4
		shuffle:     false
		num_workers: 8
		dataset:     #VALIDATION
		pin_memory:  true
	}
}


#ENC_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   127.0
	},
]

#DISP_PROCS: [
	{
		"@mode":   "partial"
		"@type":   "rearrange"
		"pattern": "c x y 1 -> c x y"
	},
	{
		"@type": "divide"
		"@mode": "partial"
		value:   10.0
	},
]


#TRAINING: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		images: {
			"@type": "JointDataset"
			mode:    "vertical"
			datasets: {
				for key, dataset in #TRAIN_DATASETS {
					for z_offset in #Z_OFFSETS {
						"\(key)_z\(z_offset)": {
							"@type": "LayerDataset"
							layer: {
								"@type": "build_layer_set"
								layers: {
									src: {
										"@type": "build_cv_layer"
										path:    #SRC_ENC_PATH + key + "/bad_alignment/z\(z_offset)"
										read_procs: #ENC_PROCS
										cv_kwargs: {"cache": false},
									}
									tgt: {
										"@type": "build_cv_layer"
										path:    #TGT_ENC_PATH + key + "/tgt_enc_2023"
										read_procs: #ENC_PROCS
										cv_kwargs: {"cache": false},
									}
									displacement: {
										"@type": "build_cv_layer"
										path:    #DISP_PATH + key + "/displacements/z\(z_offset)"
										read_procs: #DISP_PROCS
										cv_kwargs: {"cache": false},
									}
								}
							}
							sample_indexer: {
								"@type": "RandomIndexer",
								inner_indexer: {
									"@type": "VolumetricNGLIndexer",
									resolution: [dataset.resolution[0] * #DS_FACTOR, dataset.resolution[1] * #DS_FACTOR, dataset.resolution[2]],
									chunk_size: [#CHUNK_XY, #CHUNK_XY, 1],
									path: "zetta-research-nico/encoder/pairwise_aligned/" + key,
								}
							}
						},
					}
				}
			}
		}
	}
}


#VALIDATION: {
	"@type": "JointDataset"
	mode:    "horizontal"
	datasets: {
		images: {
			"@type": "JointDataset"
			mode:    "vertical"
			datasets: {
				for key, dataset in #VAL_DATASETS {
					for z_offset in #Z_OFFSETS {
						"\(key)_z\(z_offset)": {
							"@type": "LayerDataset"
							layer: {
								"@type": "build_layer_set"
								layers: {
									src: {
										"@type": "build_cv_layer"
										path:    #SRC_ENC_PATH + key + "/bad_alignment/z\(z_offset)"
										read_procs: #ENC_PROCS
										cv_kwargs: {"cache": true}
									}
									tgt: {
										"@type": "build_cv_layer"
										path:    #TGT_ENC_PATH + key + "/tgt_enc_2023"
										read_procs: #ENC_PROCS
										cv_kwargs: {"cache": true}
									}
									displacement: {
										"@type": "build_cv_layer"
										path:    #DISP_PATH + key + "/displacements/z\(z_offset)"
										read_procs: #DISP_PROCS
										cv_kwargs: {"cache": true}
									}
								}
							}
							sample_indexer: {
								"@type": "LoopIndexer",
								desired_num_samples: 100
								inner_indexer: {
									"@type": "VolumetricNGLIndexer",
									resolution: [dataset.resolution[0] * #DS_FACTOR, dataset.resolution[1] * #DS_FACTOR, dataset.resolution[2]],
									chunk_size: [#CHUNK_XY, #CHUNK_XY, 1],
									path: "zetta-research-nico/encoder/pairwise_aligned/" + key,
								}
							}
						},
					}
				}
			}
		}
	}
}


"@type": "lightning_train"
regime: #TARGET.regime
trainer: #TARGET.trainer
train_dataloader: #TARGET.train_dataloader
val_dataloader: #TARGET.val_dataloader
cluster_name: "zutils-x3"
cluster_region: "us-east1"
cluster_project: "zetta-research"
image:   "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231129"
resource_limits: {"nvidia.com/gpu": "1"}
resource_requests: {"memory": "8560Mi", "cpu": 7}
num_nodes: 1
follow_logs: false
env_vars: {"LOGLEVEL": "INFO", "NCCL_SOCKET_IFNAME": "eth0"}
local_run: false