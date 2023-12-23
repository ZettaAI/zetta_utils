#IMG_PATH: "precomputed://gs://dkronauer-ant-001-raw/brain"
#IMG_RES: [4, 4, 42]
#IMG_SIZE: [102400, 96256, 6112]

#XY_ENC_RES: _ | *[32, 64, 128, 256, 512, 1024, 2048, 4096]

#TEST_SMALL: true
#TEST_LOCAL: true
#CLUSTER_NUM_WORKERS: 8

if #TEST_SMALL {
    let bbox = #BBOX_TMPL & {start_coord: [11*4096, 5*4096, 50]
                              // end_coord:   [18*4096, 10*4096, 104]
                              end_coord:   [18*4096, 10*4096, 51]
    }
    #ENC_FLOW_TMPL: project_folder: "gs://dkronauer-ant-001-alignment/test-231221-z50-103"

    // #XY_ENC_RES: [32, 64, 128, 256, 512]
    #XY_ENC_RES: [32, 64, 128, 256, 512, 1024]
    // #XY_ENC_RES: [1024]
    // #XY_ENC_RES: [64]
    // #XY_ENC_RES: [64, 128]

    #MODELS: #CNS_MODELS
    // #ENC_FLOW_TMPL: encoding_kwargs: dst_path: "encodings_cns"
    #ENC_FLOW_TMPL: encoding_kwargs: dst_path: "encodings_cns-test-2"
    // #MODELS: #GENERAL_ENC_MODELS_V1
    // #ENC_FLOW_TMPL: encoding_kwargs: dst_path: "encodings_general_v1"
    // #MODELS: #GENERAL_ENC_MODELS_V2
    // #ENC_FLOW_TMPL: encoding_kwargs: dst_path: "encodings_general_v2"

    // Reduce chunk size for cutouts
    #ENC_FLOW_TMPL: encoding_kwargs: processing_chunk_sizes: [[512, 512, 1], [512, 512, 1]]
    // #ENC_FLOW_TMPL: defect_configs: processing_chunk_sizes: [[1024, 1024, 1], [1024, 1024, 1]]

    // Avoid unaligned writes for testing cutouts
    #ENC_FLOW_TMPL: encoding_kwargs: subchunkable_kwargs: {
        skip_intermediaries: false
        level_intermediaries_dirs: [null, "file://~/.zetta_utils/tmp/"]
    }
    #BBOX_LIST: [bbox]
}

#ENCODING_CONFIG_MODEL_TMPL: {
    path: _
    res_change_mult: _
    dst_resolution: _
    max_processing_chunk_size?: _
    fn_kwargs?: _
    op_kwargs?: _
    subchunkable_kwargs?: _
}

#MODELS: _
#ENC_FLOW_TMPL: {
    "@type": "build_pairwise_alignment_flow"
    bbox: _
    src_image_path: #IMG_PATH
    project_folder?: _

    run_encoding: true
    // run_encoding: false
    encoding_kwargs: {
        dst_path?: _
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [4096, 4096, 1]]
        crop_pad: _ | *[16, 16, 0]
        models: [
            for xy in #XY_ENC_RES {
                let model = #MODELS["\(xy)"]
                #ENCODING_CONFIG_MODEL_TMPL & {
                    path: model.path
                    res_change_mult: model.res_change_mult
                    dst_resolution: model.dst_resolution
                    max_processing_chunk_size: model.max_processing_chunk_size
                }
            }
        ]
    }

    // run_defect: true
    // defect_configs: {
    //     model_path: "gs://zetta_lee_fly_cns_001_models/jit/20221114-defects-step50000.static-1.11.0.jit"
    //     model_resolution: [64, 64, #IMG_RES[2]]
    //     processing_chunk_sizes: _ | *[[4096, 4096, 1], [4096, 4096, 1]]
    //     binarize_mask: true
    //     binarize_params: {
    //         read_threshold: 100
    //         kornia_opening_width: 11
    //         kornia_dilation_width: 3
    //         filter_cc_threshold: 320
    //         kornia_closing_width: 25
    //     }
    //     downsample_binarized_mask: true
    // }
}

#BBOX_TMPL: {
    "@type":  "BBox3D.from_coords"
    start_coord: _
    end_coord: _
    resolution: #IMG_RES
}
#BBOX_LIST: _ | *{#BBOX_TMPL & {start_coord: [0, 0, 0]
                                 end_coord:   #IMG_SIZE}}

#TEST_LOCAL: _ | *false
#TOP_LEVEL_FLOW: _ | *#GCP_FLOW
if #TEST_LOCAL {
    #TOP_LEVEL_FLOW: #LOCAL_FLOW
    // #TOP_LEVEL_FLOW: #GCP_FLOW
}
#LOCAL_FLOW: {
    "@type":      "mazepa.execute_locally"
    num_procs: 4
    semaphores_spec: {
        read: num_procs
        write: num_procs
        cuda: 2
        cpu: num_procs
    }
    target: _
}
#GCP_FLOW: {
    "@type":      "mazepa.execute_on_gcp_with_sqs"
    worker_cluster_region:  "us-east1"

    worker_cluster_project: "zetta-research"
    worker_cluster_name:    "zutils-x3"
    // worker_image:           "us.gcr.io/zetta-research/zetta_utils:tri-test-230829"
    worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:tri-231221-ant-test-schema-1"

    worker_resources: {
        memory: "21000Mi"       // sized for n1-highmem-4
        "nvidia.com/gpu": "1"
    }
    worker_replicas:     #CLUSTER_NUM_WORKERS
    local_test:          #TEST_LOCAL
    target: _
}
#TOP_LEVEL_FLOW & {
    target: #ENC_FLOW_TMPL & {bbox: #BBOX_LIST[0]}
}

#GENERAL_ENC_MODELS_V2: {
    "32": {
        path: "gs://alignment_models/general_encoders_2023/32_32_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096, 1]
    },
    "64": {
        path: "gs://alignment_models/general_encoders_2023/32_64_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048, 1]
    },
    "128": {
        path: "gs://alignment_models/general_encoders_2023/32_128_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024, 1]
    },
    "256": {
        path: "gs://alignment_models/general_encoders_2023/32_256_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512, 1]
    },
    "512": {
        path: "gs://alignment_models/general_encoders_2023/32_512_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256, 1]
    },
}

#GENERAL_ENC_MODELS_V1: {
    "32": {
        path: "gs://alignment_models/general_encoders_2023/32_32/last.static-2.0.1-model.jit"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096, 1]
    },
    "64": {
        path: "gs://alignment_models/general_encoders_2023/32_64/last.static-2.0.1-model.jit"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048, 1]
    },
    "128": {
        path: "gs://alignment_models/general_encoders_2023/32_128/last.static-2.0.1-model.jit"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024, 1]
    },
    "256": {
        path: "gs://alignment_models/general_encoders_2023/32_256_C1/last.static-2.0.1-model.jit"
        // path: "gs://alignment_models/general_encoders_2023/32_256_C2/last.static-2.0.1-model.jit"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512, 1]
    },
    "512": {
        path: "gs://alignment_models/general_encoders_2023/32_512_C1/last.static-2.0.1-model.jit"
        // path: "gs://alignment_models/general_encoders_2023/32_512_C2/last.static-2.0.1-model.jit"
        // path: "gs://alignment_models/general_encoders_2023/32_512_C3/last.static-2.0.1-model.jit"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256, 1]
    },
}

#CNS_MODELS: {
    // Adapted from https://github.com/ZettaAI/zetta_utils/blob/nkem/zfish-enc/specs/nico/inference/cns/0_encoding_cns/CNS_encoding_pyramid.cue#L47-L88
    "32": {
        path: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096, 1]
    },
    "64": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M4_conv1_unet3_lr0.0001_equi0.5_post1.6_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048, 1]
    },
    "128": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M5_conv2_unet2_lr0.0001_equi0.5_post1.4_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024, 1]
    },
    "256": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/4_M3_M6_conv3_unet1_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512, 1]
    },
    "512": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M3_M7_conv4_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/epoch=0-step=1584-backup.ckpt.model.spec.json"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256, 1]
    },
    "1024": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M3_M8_conv5_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [32, 32, 1] // 5 3-8: 32-1024
        dst_resolution: [1024, 1024, #IMG_RES[2]]
        max_processing_chunk_size: [128, 128, 1]
    }
}
