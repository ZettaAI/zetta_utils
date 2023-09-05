# pylint: skip-file
from __future__ import annotations

if __name__ == "__main__":
    import os

    from zetta_utils.api.v0 import *
    from zetta_utils.parsing import json
    from zetta_utils.training.lightning.train import _parse_spec_and_train

    LR = 2e-4
    L1_WEIGHT_START_VAL = 0.0
    L1_WEIGHT_END_VAL = 0.05
    L1_WEIGHT_START_EPOCH = 1
    L1_WEIGHT_END_EPOCH = 6
    LOCALITY_WEIGHT = 1.0
    SIMILARITY_WEIGHT = 0.0
    CHUNK_SIZE = 1024
    FM = 32
    DS_FACTOR = 16
    CHANNELS = 3
    EXP_NAME = "general_coarsener_loss"
    TRAINING_ROOT = "gs://zetta-research-nico/training_artifacts"

    EXP_VERSION = f"1.0.3_M3_M7_C{CHANNELS}_lr{LR}_locality{LOCALITY_WEIGHT}_similarity{SIMILARITY_WEIGHT}_l1{L1_WEIGHT_START_VAL}-{L1_WEIGHT_END_VAL}_N1x4"

    START_EXP_VERSION = f"1.0.9_M3_M3_unet_fm32-256_lr0.0004_locality1.0_similarity0.0_l10.0-0.05_N4xB2"
    MODEL_CKPT = None # f"gs://zetta-research-nico/training_artifacts/general_encoder_loss/{START_EXP_VERSION}/last.ckpt"

    BASE_PATH = "gs://zetta-research-nico/encoder/"

    VAL_DSET_NAME = "microns_basil"
    VALIDATION_SRC_PATH = BASE_PATH + "datasets/" + VAL_DSET_NAME
    VALIDATION_TGT_PATH = BASE_PATH + "pairwise_aligned/" + VAL_DSET_NAME + "/warped_img"

    SOURCE_PATHS = {
        "microns_pinky": {"contiguous": True, "resolution": [32, 32, 40], "num_samples": 5019},
        # "microns_basil": {"contiguous": True, "resolution": [32, 32, 40], "num_samples": 2591},
        "microns_minnie": {"contiguous": False, "resolution": [32, 32, 40], "num_samples": 2882},
        "microns_interneuron": {"contiguous": False, "resolution": [32, 32, 40], "num_samples": 6923},
        "aibs_v1dd": {"contiguous": False, "resolution": [38.8, 38.8, 45], "num_samples": 5805},
        "kim_n2da": {"contiguous": True, "resolution": [32, 32, 50], "num_samples": 446},
        "kim_pfc2022": {"contiguous": True, "resolution": [32, 32, 40], "num_samples": 3699},
        "kronauer_cra9": {"contiguous": True, "resolution": [32, 32, 42], "num_samples": 740},
        "kubota_001": {"contiguous": True, "resolution": [40, 40, 40], "num_samples": 4744},
        "lee_fanc": {"contiguous": False, "resolution": [34.4, 34.4, 45], "num_samples": 1605},
        "lee_banc": {"contiguous": False, "resolution": [32, 32, 45], "num_samples": 742},
        "lee_ppc": {"contiguous": True, "resolution": [32, 32, 40], "num_samples": 7219},
        "lee_mosquito": {"contiguous": False, "resolution": [32, 32, 40], "num_samples": 1964},
        "lichtman_zebrafish": {"contiguous": False, "resolution": [32, 32, 30], "num_samples": 2799},
        "prieto_godino_larva": {"contiguous": True, "resolution": [32, 32, 32], "num_samples": 4584},
        "fafb_v15": {"contiguous": False, "resolution": [32, 32, 40], "num_samples": 1795},
        "lichtman_h01": {"contiguous": False, "resolution": [32, 32, 33], "num_samples": 6624},
        "janelia_hemibrain": {"contiguous": True, "resolution": [32, 32, 32], "num_samples": 5304},
        "janelia_manc": {"contiguous": False, "resolution": [32, 32, 32], "num_samples": 2398},
        "nguyen_thomas_2022": {"contiguous": True, "resolution": [32, 32, 40], "num_samples": 1847},
        "mulcahy_2022_16h": {"contiguous": True, "resolution": [32, 32, 30], "num_samples": 3379},
        "wildenberg_2021_vta_dat12a": {"contiguous": True, "resolution": [32, 32, 40], "num_samples": 1704},
        "bumbarber_2013": {"contiguous": True, "resolution": [31.2, 31.2, 50], "num_samples": 7325},
        "wilson_2019_p3": {"contiguous": True, "resolution": [32, 32, 30], "num_samples": 2092},
        "ishibashi_2021_em1": {"contiguous": True, "resolution": [32, 32, 32], "num_samples": 141},
        # "ishibashi_2021_em2": {"contiguous": True, "resolution": [32, 32, 32], "num_samples": 166},
        "templier_2019_wafer1": {"contiguous": True, "resolution": [32, 32, 50], "num_samples": 5401},
        # "templier_2019_wafer3": {"contiguous": True, "resolution": [32, 32, 50], "num_samples": 3577},
        "lichtman_octopus2022": {"contiguous": True, "resolution": [32, 32, 30], "num_samples": 5673},
    }

    val_img_aug = [
        # {"@type": "to_uint8", "@mode": "partial"},
        # {
        #     "@type": "imgaug_readproc",
        #     "@mode": "partial",
        #     "augmenters": [
        #         {
        #             "@type": "imgaug.augmenters.imgcorruptlike.DefocusBlur",
        #             "severity": 1,
        #         }
        #     ],
        # },
        {"@type": "rearrange", "@mode": "partial", "pattern": "c x y 1 -> c x y"},
        {"@type": "divide", "@mode": "partial", "value": 255.0},
    ]

    train_img_aug = [
        {"@type": "divide", "@mode": "partial", "value": 255.0},
        {
            "@type": "square_tile_pattern_aug",
            "@mode": "partial",
            "prob": 0.5,
            "tile_size": {"@type": "uniform_distr", "low": 64, "high": 1024},
            "tile_stride": {"@type": "uniform_distr", "low": 64, "high": 1024},
            "max_brightness_change": {"@type": "uniform_distr", "low": 0.0, "high": 0.3},
            "rotation_degree": {"@type": "uniform_distr", "low": 0, "high": 90},
            "preserve_data_val": 0.0,
            "repeats": 1,
            "device": "cpu",
        },
        {"@type": "torch.clamp", "@mode": "partial", "min": 0.0, "max": 1.0},
        {"@type": "multiply", "@mode": "partial", "value": 255.0},
        {"@type": "to_uint8", "@mode": "partial"},
        {
            "@type": "imgaug_readproc",
            "@mode": "partial",
            "augmenters": [
                {
                    "@type": "imgaug.augmenters.SomeOf",
                    "n": 3,
                    "children": [
                        {
                            "@type": "imgaug.augmenters.OneOf",
                            "children": [
                                {
                                    "@type": "imgaug.augmenters.OneOf",
                                    "children": [
                                        {
                                            "@type": "imgaug.augmenters.GammaContrast",
                                            "gamma": (0.5, 2.0),
                                        },
                                        {
                                            "@type": "imgaug.augmenters.SigmoidContrast",
                                            "gain": (4, 6),
                                            "cutoff": (0.3, 0.7),
                                        },
                                        {
                                            "@type": "imgaug.augmenters.LogContrast",
                                            "gain": (0.7, 1.3),
                                        },
                                        {
                                            "@type": "imgaug.augmenters.LinearContrast",
                                            "alpha": (0.4, 1.6),
                                        },
                                    ],
                                },
                                {
                                    "@type": "imgaug.augmenters.AllChannelsCLAHE",
                                    "clip_limit": (0.1, 8.0),
                                    "tile_grid_size_px": (3, 64),
                                },
                            ],
                        },
                        {
                            "@type": "imgaug.augmenters.Add",
                            "value": (-40, 40),
                        },
                        {
                            "@type": "imgaug.augmenters.Sometimes",
                            "p": 1.0,
                            "then_list": [{
                                "@type": "imgaug.augmenters.imgcorruptlike.DefocusBlur",
                                "severity": 1,
                            }]
                        },
                        {
                            "@type": "imgaug.augmenters.Cutout",
                            "nb_iterations": 1,
                            "size": (0.02, 0.8),
                            "cval": (0, 255),
                            "squared": False,
                        },
                        {
                            "@type": "imgaug.augmenters.JpegCompression",
                            "compression": (0, 35),
                        },
                    ],
                    "random_order": True,
                },
            ],
        },
    ]

    shared_train_img_aug = [
        {
            "@type": "imgaug_readproc",
            "@mode": "partial",
            "augmenters": [
                {
                    "@type": "imgaug.augmenters.Sequential",
                    "children": [
                        {
                            "@type": "imgaug.augmenters.Rot90",
                            "k": [0, 1, 2, 3],
                        },
                        {
                            "@type": "imgaug.augmenters.Fliplr",
                            "p": 0.25,
                        },
                        {
                            "@type": "imgaug.augmenters.Flipud",
                            "p": 0.25,
                        },
                    ],
                    "random_order": True,
                }
            ],
        },
        {"@type": "rearrange", "@mode": "partial", "pattern": "c x y 1 -> c x y"},
        {"@type": "divide", "@mode": "partial", "value": 255.0},
    ]


    training = {
        "@type": "JointDataset",
        "mode": "horizontal",
        "datasets": {
            "images": {
                "@type": "JointDataset",
                "mode": "vertical",
                "datasets": {
                    name: {
                        "@type": "LayerDataset",
                        "layer": {
                            "@type": "build_layer_set",
                            "layers": {
                                "src_img": {
                                    "@type": "build_cv_layer",
                                    "path": BASE_PATH + "datasets/" + name,
                                    "read_procs": train_img_aug,
                                    "cv_kwargs": {"cache": False},
                                },
                                "tgt_img": {
                                    "@type": "build_cv_layer",
                                    "path": BASE_PATH + "pairwise_aligned/" + name + "/warped_img",
                                    "read_procs": train_img_aug,
                                    "cv_kwargs": {"cache": False},
                                },
                            },
                            "readonly": True,
                            "read_procs": shared_train_img_aug,
                        },
                        "sample_indexer": {
                            "@type": "RandomIndexer",
                            "inner_indexer": {
                                "@type": "VolumetricNGLIndexer",
                                "resolution": settings["resolution"],
                                "chunk_size": [CHUNK_SIZE, CHUNK_SIZE, 1],
                                "path": "zetta-research-nico/encoder/pairwise_aligned/" + name,
                            }
                        }
                    }
                    for name, settings in SOURCE_PATHS.items()
                },
            },
            "field": {
                "@type": "LayerDataset",
                "layer": {
                    "@type": "build_cv_layer",
                    "path": "gs://zetta-research-nico/perlin_noise_fields/1px",
                    "read_procs": [
                        {"@type": "rearrange", "@mode": "partial", "pattern": "c x y 1 -> c x y"},
                    ],
                    "cv_kwargs": {"cache": False},
                },
                "sample_indexer": {
                    "@type": "RandomIndexer",
                    "inner_indexer": {
                        "@type": "VolumetricStridedIndexer",
                        "bbox": {
                            "@type": "BBox3D.from_coords",
                            "start_coord": [0, 0, 0],
                            "end_coord": [2048, 2048, 2040],
                            "resolution": [4, 4, 45],
                        },
                        "stride": [128, 128, 1],
                        "chunk_size": [CHUNK_SIZE, CHUNK_SIZE, 1],
                        "resolution": [4, 4, 45],
                    },
                },
            },
        },
    }

    validation = {
        "@type": "JointDataset",
        "mode": "horizontal",
        "datasets": {
            "images": {
                "@type": "LayerDataset",
                "layer": {
                    "@type": "build_layer_set",
                    "layers": {
                        "src_img": {
                            "@type": "build_cv_layer",
                            "path": VALIDATION_SRC_PATH,
                            "read_procs": val_img_aug,
                        },
                        "tgt_img": {
                            "@type": "build_cv_layer",
                            "path": VALIDATION_TGT_PATH,
                            "read_procs": val_img_aug,
                        },
                    },
                    "readonly": True,
                },
                "sample_indexer": {
                    "@type": "LoopIndexer",
                    "desired_num_samples": 100,
                    "inner_indexer": {
                        "@type": "VolumetricNGLIndexer",
                        "resolution": [32,32,40],
                        "chunk_size": [CHUNK_SIZE, CHUNK_SIZE, 1],
                        "path": "zetta-research-nico/encoder/pairwise_aligned/" + VAL_DSET_NAME,
                    }
                },
            },
            "field": {
                "@type": "LayerDataset",
                "layer": {
                    "@type": "build_cv_layer",
                    "path": "gs://zetta-research-nico/perlin_noise_fields/1px",
                    "read_procs": [
                        {"@type": "rearrange", "@mode": "partial", "pattern": "c x y 1 -> c x y"},
                    ],
                },
                "sample_indexer": {
                    "@type": "LoopIndexer",
                    "desired_num_samples": 100,
                    "inner_indexer": {
                        "@type": "VolumetricStridedIndexer",
                        "bbox": {
                            "@type": "BBox3D.from_coords",
                            "start_coord": [0, 0, 0],
                            "end_coord": [2048, 2048, 2040],
                            "resolution": [4, 4, 45],
                        },
                        "stride": [512, 512, 1],
                        "chunk_size": [CHUNK_SIZE, CHUNK_SIZE, 1],
                        "resolution": [4, 4, 45],
                    },
                },
            },
        },
    }


    target = BuilderPartial(
        {
            "@type": "lightning_train",
            "regime": {
                "@type": "BaseEncoderRegime",
                "@version": "0.0.2",
                "max_displacement_px": 32.0,
                "val_log_row_interval": 20,
                "train_log_row_interval": 100,
                "lr": LR,
                "l1_weight_start_val": L1_WEIGHT_START_VAL,
                "l1_weight_end_val": L1_WEIGHT_END_VAL,
                "l1_weight_start_epoch": L1_WEIGHT_START_EPOCH,
                "l1_weight_end_epoch": L1_WEIGHT_END_EPOCH,
                "locality_weight": LOCALITY_WEIGHT,
                "similarity_weight": SIMILARITY_WEIGHT,
                "ds_factor":              DS_FACTOR,
                "empty_tissue_threshold": 0.6,
                "model": {
                    "@type": "load_weights_file",
                    "ckpt_path": MODEL_CKPT,
                    "component_names": [
                        "model",
                    ],
                    "model": {
                        "@type": "torch.nn.Sequential",
                        "modules": [
                            {
                                "@type": "ConvBlock",
                                "num_channels": [1, FM],
                                "kernel_sizes": [5, 5],
                                "padding_modes": "reflect",
                                "activate_last": True,
                            },
                            {
                                "@type":     "torch.nn.MaxPool2d",
                                "kernel_size": 2
                            },
                            {
                                "@type": "ConvBlock",
                                "num_channels": [FM, FM, FM, FM*2],
                                "kernel_sizes": [3, 3],
                                "padding_modes": "reflect",
                                "activate_last": True,
                                "skips": {"0": 2},
                            },
                            {
                                "@type":     "torch.nn.MaxPool2d",
                                "kernel_size": 2
                            },
                            {
                                "@type": "ConvBlock",
                                "num_channels": [FM*2, FM*2, FM*2, FM*4],
                                "kernel_sizes": [3, 3],
                                "padding_modes": "reflect",
                                "activate_last": True,
                                "skips": {"0": 2},
                            },
                            {
                                "@type":     "torch.nn.MaxPool2d",
                                "kernel_size": 2
                            },
                            {
                                "@type": "ConvBlock",
                                "num_channels": [FM*4, FM*4, FM*4, FM*8],
                                "kernel_sizes": [3, 3],
                                "padding_modes": "reflect",
                                "activate_last": True,
                                "skips": {"0": 2},
                            },
                            {
                                "@type":     "torch.nn.MaxPool2d",
                                "kernel_size": 2
                            },
                            {
                                "@type": "ConvBlock",
                                "num_channels": [FM*8, FM*8, FM*8, FM*8],
                                "kernel_sizes": [3, 3],
                                "padding_modes": "reflect",
                                "activate_last": True,
                                "skips": {"0": 3},
                            },
                            {
                                "@type": "torch.nn.Conv2d",
                                "in_channels": FM*8,
                                "out_channels": CHANNELS,
                                "kernel_size": 1,
                            },
                            {"@type": "torch.nn.Tanh"},
                        ],
                    },
                },
            },
            "trainer": {
                "@type": "ZettaDefaultTrainer",
                "accelerator": "gpu",
                "precision": "16-mixed",
                "strategy": "auto",
                # "use_distributed_sampler": False,
                "devices": 4,
                "max_epochs": 100,
                "default_root_dir": TRAINING_ROOT,
                "experiment_name": EXP_NAME,
                "experiment_version": EXP_VERSION,
                "log_every_n_steps": 100,
                "val_check_interval": 500,
                # "limit_val_batches": 0,
                # "track_grad_norm": 2,
                # "gradient_clip_algorithm": "norm",
                # "gradient_clip_val": CLIP,
                # "detect_anomaly": True,
                # "overfit_batches": 100,
                "reload_dataloaders_every_n_epochs": 1,
                "checkpointing_kwargs": {"update_every_n_secs": 1700, "backup_every_n_secs": 3700},
            },
            "train_dataloader": {
                "@type": "TorchDataLoader",
                "batch_size": 2,
                # "shuffle": True,
                "sampler": {
                    "@type": "SamplerWrapper",
                    "sampler": {
                        "@type": "TorchRandomSampler",  # Random order across all samples and all datasets
                        "data_source": {"@type": "torch.arange", "end": sum([settings["num_samples"] for settings in SOURCE_PATHS.values()])},
                        "replacement": False,
                        "num_samples": 8000,
                    },
                },
                "num_workers": 28,
                "dataset": training,
                "pin_memory": True,
            },
            "val_dataloader": {
                "@type": "TorchDataLoader",
                "batch_size": 1,
                "shuffle": False,
                "num_workers": 28,
                "dataset": validation,
                "pin_memory": True,
            },
        }
    )


    os.environ["ZETTA_RUN_SPEC"] = json.dumps(target.spec)

    # _parse_spec_and_train()

    lightning_train_remote(
        worker_cluster_name="zutils-x3",
        worker_cluster_region="us-east1",
        worker_cluster_project="zetta-research",
        worker_image="us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20231106",
        worker_resources={"nvidia.com/gpu": "4"},
        worker_resource_requests={"memory": "27560Mi", "cpu": 28},
        num_nodes=1,
        spec_path=target,
        follow_logs=False,
        env_vars={"LOGLEVEL": "INFO", "NCCL_SOCKET_IFNAME": "eth0"},
    )
