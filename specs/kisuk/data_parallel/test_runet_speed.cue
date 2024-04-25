#EXP_NAME:      "test_data_parallel"
#TRAINING_ROOT: "gs://zetta-research-kisuk/zetta_utils/training_artifacts"
#LR:            1e-3
#CLIP:          0e-5
#CHUNK_SIZE:  	[92, 92, 92]
#PAD_SIZE:		[18, 18, 18]
#MODEL_CKPT:  	null
#EXP_VERSION: 	"runet_affinity_gpu4_worker8_x6"

#NN_EDGES: [
	[1, 0, 0], [0, 1, 0], [0, 0, 1],
]

#LR_EDGES: [
	[ 5, 0, 0], [0,  5, 0], [0, 0,  5],
	[10, 0, 0], [0, 10, 0], [0, 0, 10],
	[15, 0, 0], [0, 15, 0], [0, 0, 15],
]

#LOSS: {
	"@type": "BinaryLossWithInverseMargin"
	criterion: {
		"@type": "torch.nn.BCEWithLogitsLoss"
		"@mode": "partial"
	}
	reduction: "mean"
	balancer: {
		"@type": "BinaryClassBalancer"
		group: 1
	}
	// margin: 0.1
}

// #LOSS: {
// 	"@type": "AffinityLoss"
//     edges: _
// 	criterion: {
// 		"@type": "torch.nn.BCEWithLogitsLoss"
// 		"@mode": "partial"
// 	}
// 	reduction: "mean"
// 	balancer: {
// 		"@type": "BinaryClassBalancer"
// 		group: 1
// 	}
// 	// margin: 0.1
// }

"@type":    "mazepa.execute_on_gcp_with_sqs"
"@version": "0.0.1"

worker_image: "us.gcr.io/zetta-research/zetta_utils:kisuk-test-dp"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "4"
}
worker_replicas:     1
batch_gap_sleep_sec: 5

local_test: false

target: {
	"@type": "lightning_train"
	"@mode": "partial"

	regime: {
		"@type":                "BaseAffinityRegime"
		val_log_row_interval:   25
		train_log_row_interval: 100
		lr:                     #LR
		amsgrad:				true
		logits:					true

		model: {
			"@type": "load_weights_file"
			model: {
				"@type": "torch.nn.Sequential"
				modules: [
					{
						"@type": "torch.nn.Conv3d"
						in_channels: 1
						out_channels: 16
						kernel_size: [3, 3, 3]
						padding: [0, 0, 0]
						bias: false
					},
					{
						"@type": "UNet"
						"@version": "0.0.1"
						conv: {
							"@type": "torch.nn.Conv3d"
							"@mode": "partial"
							bias: false
						}
						downsample: {
							"@type": "torch.nn.MaxPool3d"
							"@mode": "partial"
							kernel_size: [2, 2, 2]
						}
						upsample: {
							"@type": "UpConv"
							"@mode": "partial"
							kernel_size: [1, 1, 1]
							upsampler: {
								"@type": "torch.nn.Upsample"
								"@mode": "partial"
								scale_factor: [2, 2, 2]
								mode: "trilinear"
							}
							conv: {
								"@type": "torch.nn.Conv3d"
								"@mode": "partial"
								bias: false
							}
						}
						activation: {
							"@type": "torch.nn.ReLU"
							"@mode": "partial"
							inplace: true
						}
						normalization: {
							"@type": "torch.nn.InstanceNorm3d"
							"@mode": "partial"
							affine: true
						}
						list_num_channels: [
							[16, 16, 16, 16, 16],

							[16, 128, 128, 128, 128],

							[16, 16, 16, 16, 16],
						]
						kernel_sizes: [3, 3, 3]
						paddings: [0, 0, 0]
						skips: {"1": 3}
						normalize_last: true
						activate_last: true
						activation_mode: "pre"
					},
					{
						"@type": "MultiHeadedOutput"
						in_channels: 16
						heads: {
							"affinity": 3
							"long-range": 9
						}
						conv: {
							"@type": "torch.nn.Conv3d"
							"@mode": "partial"
							kernel_size: [3, 3, 3]
							padding: [0, 0, 0]
							bias: true
						},
					},
				]
			}
			ckpt_path: #MODEL_CKPT
			component_names: [
				"model",
			]
		}

		// criteria: {
		// 	"affinity":   #LOSS & {edges: #NN_EDGES}
		// 	"long-range": #LOSS & {edges: #LR_EDGES}
		// }
        criteria: {
			"affinity":   #LOSS
			"long-range": #LOSS
		}

		loss_weights: {
			"affinity":   1.0
			"long-range": 1.0
		}
	}
	trainer: {
		"@type":                 "ZettaDefaultTrainer"
		accelerator:             "gpu"
		// devices:                 1
        devices:                 4
		max_epochs:              1
		default_root_dir:        #TRAINING_ROOT
		experiment_name:         #EXP_NAME
		experiment_version:      #EXP_VERSION
		log_every_n_steps:       10
		limit_val_batches:		 0.0
		// val_check_interval:      1000
		checkpointing_kwargs: {
			update_every_n_secs: 60
			backup_every_n_secs: 900
		}
	}

	train_dataloader: {
		"@type":            "TorchDataLoader"
		batch_size:         1
		shuffle:            false
		num_workers:        8
		dataset:            #train_dset
		persistent_workers: true
	}
	val_dataloader: {
		"@type":            "TorchDataLoader"
		batch_size:         1
		shuffle:            false
		num_workers:        4
		dataset:            #val_dset
		persistent_workers: true
	}
}

#dset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "build_layer_set"
		layers: {
			data_in: {
				"@type": "build_cv_layer"
				path:    "gs://zetta_research_datasets/zettasets/hemibrain/eb-inner/image"
				cv_kwargs: {}
				index_procs: [
					{
						"@type": "VolumetricIndexPadder"
						pad: #PAD_SIZE
					},
				]
				read_procs: [
					{"@type": "divide", "@mode": "partial", value: 255.0},
				]

			}
			target: {
				"@type": "build_cv_layer"
				path:    "gs://zetta_research_datasets/zettasets/hemibrain/eb-inner/seg/000"
				cv_kwargs: {}
				read_procs: [
					{"@type": "to_float32", "@mode": "partial"},
				]
			}
		}
		readonly: true
		read_procs: [
			{
				"@type": "ROIMaskProcessor"
				start_coord: _
				end_coord: 	 _
				resolution:  [8, 8, 8]
				targets: 	 ["target"]
			},
			{
				"@type": "AffinityProcessor"
				source: "target"
				spec: {
					"affinity":   #NN_EDGES
					"long-range": #LR_EDGES
				}
			},
            // {
            //     "@type": "MultiHeadedProcessor"
            //     spec: {
            //         "target": [
            //             "affinity",
            //             "long-range",
            //         ]
            //     }
            // },
		]
	}
	sample_indexer: {
		"@type": "RandomIndexer"
		inner_indexer: {
			"@type": "VolumetricStridedIndexer"
			resolution: [8, 8, 8]
			chunk_size: #CHUNK_SIZE
			stride:     _
			bbox: {
				"@type":     "BBox3D.from_coords"
				start_coord: _
				end_coord:   _
				resolution: [8, 8, 8]
			}
			mode: "shrink"
		}
		replacement: true
	}
}

#train_dset: #dset_settings & {
	layer: read_procs: [
		{
			start_coord: [128, 128, 128]
			end_coord: 	 [648, 648, 648]
		},
		_,
	]
	sample_indexer: inner_indexer: stride: [1, 1, 1]
	sample_indexer: inner_indexer: bbox: {
		start_coord: [128 + 18, 128 + 18, 128 + 18]
		end_coord: 	 [648 - 18, 648 - 18, 648 - 18]
	}
}

#val_dset: #dset_settings & {
	layer: read_procs: [
		{
			start_coord: [128, 128, 128]
			end_coord: 	 [648, 648, 648]
		},
		_,
	]
	sample_indexer: inner_indexer: stride: [1, 1, 1]
	sample_indexer: inner_indexer: bbox: {
		start_coord: [128 + 18, 128 + 18, 128 + 18]
		end_coord: 	 [648 - 18, 648 - 18, 648 - 18]
	}
}
