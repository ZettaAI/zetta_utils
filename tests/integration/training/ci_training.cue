import (
    L "list"
)

#EXP_NAME:     "ci_training"
#TRAINING_ROOT: "gs://zutils_integration_tests_tmp_2w/training_artifacts/ci_training"
#LR: _
#BATCH_SIZE: _ | *1
#EXP_VERSION_: "v1.0.0-\(#RESOLUTION[0])nm-\(#CONFIG_SIZE)-bs_\(#EFFECTIVE_BATCH_SIZE)-lr_\(#LR)-fov_\(#VALID_FOV_SIZE[0])"
#TRY_NUMBER: _ | *5

local_run: false
num_nodes: 2
#LR: 0.00004 * 2
#TRAINING_ITERATIONS: _ | *50
#RESOLUTION: [16, 16, 16]

trainer: max_steps: #TRAINING_ITERATIONS
regime: train_log_row_interval: 100
regime: val_log_row_interval: 500
trainer: checkpointing_kwargs: {
    update_every_n_secs: 120    // 2m
    backup_every_n_secs: 180    // 3m
}

//required_zones: ["us-central1-b"]

////////////////////////////////////////////////////////////////
// --> layer keys need the appropriate suffix for imgaug to work
////////////////////////////////////////////////////////////////
#IMG_IN_KEY: "raw_img"
#SEG_KEY: "in_seg"
#OUTPUT_AFFS_KEY: #SEG_KEY

///////////////////////////////////////////////////////////////////
////////////////////////// DDP Config /////////////////////////////
///////////////////////////////////////////////////////////////////

"@type": "lightning_train"
local_run: _ | *false
cluster_project: "zetta-research"
cluster_name:    "zutils-x3"
cluster_region:  "us-east1"
image: "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:PYTHON_VERSION_github_action_integration_test"

gpu_accelerator_type: "nvidia-l4"
#NUM_DATALOADER_WORKERS: _ | *resource_requests.cpu
// #NUM_DATALOADER_WORKERS: 15
resource_requests: {
	// https://cloud.google.com/compute/docs/gpus#l4-gpus
	// 4/16, 8/32, 12/48, 16/64, 32/128
	cpu: _ | *11
    cpu: 15
	memory: "40000Mi"
}
resource_limits: {
	"nvidia.com/gpu": trainer.devices
}
env_vars: {
	"LOGLEVEL":             "INFO"
	"NCCL_SOCKET_IFNAME":   "eth0"
}
num_nodes: _ | *1
follow_logs: _ | *true
follow_logs: false  // buggy follow_logs auto exits after 4 hours
provisioning_model: "standard"

trainer: precision: "bf16-mixed"

#CONFIG_SIZE: "small3"
#N_FMAPS: _
#N_CONVS: _
#SKIP_CONNECTIONS: _
#N_FMAPS: [16, 32, 64, 128]  // 128, 64, 32, 16, 8
#N_CONVS: 4
#SKIP_CONNECTIONS: {
    "0": 2,
    "2": 4,
}

#VALID_FOV_SIZE:    _
#LOSS_CROP_PAD:     _
#AUG_PAD:           _
#VALID_FOV_SIZE:    [96, 96, 96]
#LOSS_CROP_PAD:		[8, 8, 8]
#AUG_PAD:           [32, 32, 32]

#PATHS_CUTOUT_TEMPLATE: {
    _img_path: "gs://zutils_integration_tests/hemibrain/emdata/clahe_yz/jpeg"
    _cutout_label: _
    _seg_path: "gs://zutils_integration_tests/hemibrain/v1.0/segmentation"
    ...
}
#PATHS_CUTOUT_001: #PATHS_CUTOUT_TEMPLATE & {
    _cutout_label: "cutout1"
}
#LABELS: {
    cutout1: #PATHS_CUTOUT_001 & {
        _bound_resolution: [8, 8, 8]
        _start: [21504, 24576, 19456]
        _end:   [for x in _start {x+1024}]
        _weight: 100_000_000
    }
}
#TRAINING_DATASETS: { for key, value in #LABELS { "\(key)": #DSET_SETTINGS & value & {_train: true} } }
#TRAINING: {
    "@type": "JointDataset"
    mode: "vertical"
    datasets: {
        for key, value in #TRAINING_DATASETS {
            "\(key)": value & {_weight: _ | *10_000_000}
        }
    }
}
let val_coords = [
    [16384, 16384, 16384],
]
#VAL_LABELS: [
    for coord in val_coords {
        #PATHS_CUTOUT_001 & {
            _start: coord
            _end: [for i, x in _start {x+#VALID_FOV_SIZE[i]*2}]
            _bound_resolution: [8, 8, 8]
        }
    }
]
#VAL_DATASETS: [ for label in #VAL_LABELS { #DSET_SETTINGS & label & {_train: false} } ]
#VALIDATION: {
    "@type": "JointDataset"
    mode: "vertical"
    datasets: {
        "cutout1": #VAL_DATASETS[0] & {_weight: 1}
    }
}

#N_DOWNSAMPLES: _
#MODEL: {
    "@type": "torch.nn.Sequential"
    modules: [
        {
            "@type": "torch.nn.Conv3d"
            in_channels: 1
            out_channels: #N_FMAPS[0]
            kernel_size: [1, 1, 1]
            padding: "same"
            bias: true
        },
        {
            "@type": "UNet"
            "@version": "0.0.2"
            conv: {
                "@type": "torch.nn.Conv3d"
                "@mode": "partial"
                bias: true
            }
            downsample: {
                "@type": "torch.nn.Conv3d"
                "@mode": "partial"
                kernel_size: [2, 2, 2]
                stride: [2, 2, 2],
                bias: false
            }
            upsample: {
                "@type": "torch.nn.ConvTranspose3d"
                "@mode": "partial"
                kernel_size: [2, 2, 2],
                stride: [2, 2, 2],
                bias: false
            }
            activation: {
                "@type": "torch.nn.ReLU"
                "@mode": "partial"
                inplace: true
            }
            list_num_channels: [
                // down & middle convs
                for i, n in #N_FMAPS {
                    [for j in L.Range(0, #N_CONVS+1, 1) {#N_FMAPS[i]}]
                }
                // up convs
                for i in L.Range(len(#N_FMAPS)-2, -1, -1) {
                    [for j in L.Range(0, #N_CONVS+1, 1) {#N_FMAPS[i]}]
                }
            ]
            kernel_sizes: [3, 3, 3]
            paddings: "same"
            padding_modes: "zeros"
            unet_skip_mode: "sum"  // sum or concat
            skips: #SKIP_CONNECTIONS
            activate_last: true
            activation_mode: "post"
        },
        {
            "@type": "torch.nn.Conv3d"
            in_channels: #N_FMAPS[0],
            out_channels: 3,
            kernel_size: [1, 1, 1]
            padding: "same"
            bias: true
        },
    ]
}

#BALANCE_MODE: _ | *"none"
#BALANCE_CLIP_FACTOR: _ | *0.1
#MEAN_REDUCTION_CLIP_FACTOR: _ | *0.1
#OPTIMIZER: _ | *"RAdam"
#WEIGHT_DECAY: _ | *0.01
#LR_SCHEDULER: _ | *null
#CKPT_PATH: _ | *null

#EFFECTIVE_BATCH_SIZE: trainer.devices*num_nodes*#BATCH_SIZE
regime: {
    "@type": "BinarySupervisedRegime"
    optimizer: {
        "@type": "torch.optim.\(#OPTIMIZER)"
        "@mode": "partial"
        if #OPTIMIZER == "RAdam" {
            decoupled_weight_decay: true
            weight_decay: #WEIGHT_DECAY * #EFFECTIVE_BATCH_SIZE
        }
    }
    lr: #LR
    logits: true
    min_nonz_frac:              0.0
    mean_reduction_clip:        #MEAN_REDUCTION_CLIP_FACTOR
    reduction_mode: "mean"

    model: {
        "@type": "load_weights_file"
        model: #MODEL,
        ckpt_path: _ | *null
        if #CKPT_PATH != null {
            ckpt_path: #CKPT_PATH
        }
        component_names: [
            "model",
        ]
    }
    train_log_row_interval: _ | *500
    val_log_row_interval?: _
    loss_crop_pad: #LOSS_CROP_PAD
    log_max_dims: [120, 120, 9]

    data_in_key: #IMG_IN_KEY
    target_key: #OUTPUT_AFFS_KEY
}
trainer: {
    "@type":                 "ZettaDefaultTrainer"
    accelerator:             _ | *"auto"
    precision:               _ | *"16-mixed",
    strategy:                "auto",
    devices:                 _ | *1
    num_nodes:          	 1
    max_epochs:              -1
    max_steps:				 _
    default_root_dir:        #TRAINING_ROOT
    experiment_name:         #EXP_NAME
    experiment_version:      #EXP_VERSION
    log_every_n_steps:       100
    // gc_interval: 200

    num_sanity_val_steps:	0
    val_check_interval:      _ | *2000

    progress_bar_kwargs: {
        leave: true
    }
    checkpointing_kwargs: {
        update_every_n_secs: _ | *600    // update latest checkpoint every 10m
        backup_every_n_secs: _ | *900    // save backup copies every 15m
        // update_every_n_secs: null
        // backup_every_n_secs: null
        // update_every_n_steps: _ | *2000
        // backup_every_n_steps: _ | *10000
    }
    profiler: "simple"
    benchmark?: _
}
#DEBUG_DATALOADER: _ | *false
train_dataloader: {
    "@type":            "TorchDataLoader"
    batch_size:         #BATCH_SIZE
    shuffle:            true
    num_workers:        _ | *#NUM_DATALOADER_WORKERS
    dataset:            #TRAINING
    pin_memory:			true
    persistent_workers: _ | *true
    if #DEBUG_DATALOADER {
        num_workers:        0
        persistent_workers: false
    }
}
val_dataloader: _ | *{
    "@type":            "TorchDataLoader"
    batch_size:         _ | *#BATCH_SIZE
    shuffle:            false
    num_workers:        _ | *4
    dataset:            #VALIDATION
    pin_memory:			true
    persistent_workers: _ | *true
    if #DEBUG_DATALOADER {
        num_workers:        0
        persistent_workers: false
    }
}

#BUILD_LAYER: {
    "@type": "build_cv_layer"
    cv_kwargs: {
        cache: true         // enable disk cache
    }
    cache_bytes_limit: 0    // disable in-mem cache
    ...
}

#DSET_SETTINGS: {
    _img_path: _
    _label_path: _
    _seg_path: _
    _start: _
    _end: _
    _bound_resolution: _
    _train: _ | *false | bool
    _weight: _ | *null

    "@type": "LayerDataset"
    layer: {
        "@type": "build_layer_set"
        layers: {
            "\(#IMG_IN_KEY)": #BUILD_LAYER & {
                path: _img_path
                index_procs: [#PAD_INDEX_PROC]
            }
            "\(#SEG_KEY)": #BUILD_LAYER & {
                path: _seg_path
                index_procs: [#PAD_INDEX_PROC]
                read_procs: [
                    // {"@type": "lambda", lambda_str: "lambda x: fastremap.renumber(x, preserve_zero=True)[0]"},
                    {"@type": "lambda", lambda_str: "lambda x: __import__(\"fastremap\").renumber(x, preserve_zero=True)[0]"},
                ]
            }
        }
        readonly: true
        read_procs: [
            if (_train == true) && (#RESOLUTION[0] == #RESOLUTION[2]) {
                {
                    "@type": "transpose_3d_augmentation", "@mode": "partial",
                    prob: 0.666,
                    axes: [
                        // {'__tuple__' : [0, 1, 2, 3]},
                        {'__tuple__' : [0, 3, 2, 1]},
                        {'__tuple__' : [0, 1, 3, 2]},
                    ]
                }
            }

            if _train == true {
                {
                    "@type": "imgaug_readproc"
                    "@mode": "partial"
                    mode: "3d"
                    "augmenters": [
                        {
                            "@type": "imgaug.augmenters.Fliplr"
                            "p": 0.5
                        },
                        {
                            "@type": "imgaug.augmenters.Affine"
                            scale: {
                                x: {'__tuple__' : [0.5, 2.0]}
                                y: {'__tuple__' : [0.5, 2.0]}
                            }
                            rotate: {'__tuple__' : [-180, 180]}
                            shear: {'__tuple__' : [-30, 30]}
                        },
                    ]
                },
            }
            {
                "@type": "crop", "@mode": "partial",
                crop: #AUG_PAD
            },
            if _train == true {
                {
                    "@type": "imgaug_readproc"
                    "@mode": "partial"
                    mode: "3d"
                    targets: [#IMG_IN_KEY]
                    "augmenters": [
                        {
                            // Whole vol brightness
                            "@type": "imgaug.augmenters.Add"
                            "value": {'__tuple__' : [-40, 40]}
                        },
                        {
                            // Whole vol noise level
                            "@type": "imgaug.augmenters.AdditiveGaussianNoise"
                            scale: {'__tuple__' : [0, 15]}
                        },
                    ]
                }
            },
            if _train == true {
                {
                    "@type": "imgaug_readproc"
                    "@mode": "partial"
                    targets: [#IMG_IN_KEY]
                    "augmenters": [
                        {
                            // Per section brightness changes
                            "@type": "imgaug.augmenters.Add"
                            "value": {'__tuple__' : [-30, 30]}
                        },
                        {
                            // Per section random noise
                            "@type": "imgaug.augmenters.AdditiveGaussianNoise"
                            scale: {'__tuple__' : [0, 15]}
                        },
                        {
                            // Per section pixel blackout
                            "@type": "imgaug.augmenters.CoarseDropout"
                            p: 0.025
                            size_percent: {'__tuple__' : [0.02, 0.25]}  // squares of 4 - 50px
                        },
                    ]
                }
            },
            {
                "@type": "fastmorph_erode_3d", "@mode": "partial"
                targets: [#SEG_KEY], iterations: 1
            },
            {
                "@type": "seg_to_aff_3d"
                "@mode": "partial"
                edges: [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
                same_padding: true
                targets: [#OUTPUT_AFFS_KEY]
            },

            // Convert uint8 image to float32 for training. Performed post augs to improve performance
            {"@type": "to_float32", "@mode": "partial", targets: [#IMG_IN_KEY]},
            {"@type": "multiply", "@mode": "partial", value: 1.0/255.0, targets: [#IMG_IN_KEY]},

            {
                "@type": "dataloader_gc_collect"
                "@mode": "partial"
                every_n: 100
            },
        ]
    }
    if _train == true {
        sample_indexer: {
            "@type": "LoopIndexer"
            inner_indexer: {
                "@type": "RandomIndexer"
                inner_indexer: {
                    "@type": "VolumetricStridedIndexer"
                    resolution: #RESOLUTION
                    chunk_size: #VALID_FOV_SIZE
                    stride:     _ | *[1, 1, 1]
                    bbox: {
                        "@type": "BBox3D.from_coords"
                        start_coord:    _start
                        end_coord:      _end
                        resolution:     _bound_resolution
                    }
                    mode: "shrink"
                }
                replacement: true
            }
            desired_num_samples: _weight
        }
    }
    if _train == false {
        sample_indexer: {
            "@type": "VolumetricStridedIndexer"
            resolution: #RESOLUTION
            chunk_size: #VALID_FOV_SIZE
            stride:     #VALID_FOV_SIZE
            bbox: {
                "@type": "BBox3D.from_coords"
                start_coord:    _start
                end_coord:      _end
                resolution:     _bound_resolution
            }
            mode: "shrink"
        }
    }
}

#PAD_INDEX_PROC: {
    // sampler is only picking "valid" patches - adding pads
    "@type": "VolumetricIndexPadder"
    pad: [
        #LOSS_CROP_PAD[0] + #AUG_PAD[0],
        #LOSS_CROP_PAD[1] + #AUG_PAD[1],
        #LOSS_CROP_PAD[2] + #AUG_PAD[2],
    ]
}

#EXP_VERSION: (#EXP_VERSION_ + "-x\(#TRY_NUMBER)")
