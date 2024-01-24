#IMG_PATH: "precomputed://gs://dkronauer-ant-001-raw/brain"
#IMG_RES: [4, 4, 42]
#IMG_SIZE: [102400, 96256, 6112]

// #TEST_LOCAL: true
// #CLUSTER_NUM_WORKERS: 32
#CLUSTER_NUM_WORKERS: 128

#RUN_ALIGN: true
// #RUN_MISD: true

#PROJECT: "production-240118"
#PAIR_FLOW_TMPL: project_folder: _ | *"gs://dkronauer-ant-001-alignment/\(#PROJECT)"

#GCP_FLOW: num_procs: 4
#GCP_FLOW: semaphores_spec: {
    read: #GCP_FLOW.num_procs
    write: #GCP_FLOW.num_procs
    cuda: 1
    cpu: #GCP_FLOW.num_procs
}

#PAIR_FLOW_TMPL: {
    run_compute_field: #RUN_ALIGN
    compute_field_flow_kwargs: src_path: "gs://dkronauer-ant-001-encodings/\(#PROJECT)/encodings_masked"
    run_invert_field: #RUN_ALIGN
    run_warp: #RUN_ALIGN

    run_misd: #RUN_MISD
    run_enc_warped_imgs: #RUN_MISD
}


if true {
    // #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0,   0]
                          // end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 152]}}
    // #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0, 152]
    //                       end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 603]}}
    #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0, 603]
                          end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 1300]}}
    #BBOX_LIST: [#BBOX]

    #PAIR_FLOW_TMPL: compute_field_flow_kwargs: stages: [
        {dst_resolution: [512, 512, #IMG_RES[2]], fn_kwargs: {sm: 300, num_iter: 700, lr: 0.015}},
        {dst_resolution: [256, 256, #IMG_RES[2]], fn_kwargs: {sm: 150, num_iter: 700, lr: 0.030}},
        {dst_resolution: [128, 128, #IMG_RES[2]], fn_kwargs: {sm: 100, num_iter: 500, lr: 0.050}},
        {dst_resolution: [ 64,  64, #IMG_RES[2]], fn_kwargs: {sm:  50, num_iter: 300, lr: 0.100}},
        {dst_resolution: [ 32,  32, #IMG_RES[2]], fn_kwargs: {sm:  25, num_iter: 200, lr: 0.100}},
    ]

    // volume is small enough in xy (6400 x 6016px in 64nm); going to 4096x4096 would result in wasted work
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    #PAIR_FLOW_TMPL: misd_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
}

//////////////////////////////////
// MISALIGNMENT DETECTOR CONFIG //
//////////////////////////////////
if true {
    // configs for misd-v2, z-1 model
    #MISD_MODEL_PATH: "gs://zetta-research-nico/training_artifacts/aced_misd_general/3.2.0_dsfactor2_thr1.5_lr0.0001_z1/epoch=44-step=11001-backup.ckpt.model.spec.json"
    #MISD_ENCODER: #GENERAL_ENC_MODELS_V2["64"]
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: dst_path: "imgs_warped_encoded-v2-z1"
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: reencode_tgt: {dst_path: "encodings_misd-v2-z1"}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: dst_path: "misalignments-v2-z1"
    #MISD_MODEL: {fn: {"@type": "MisalignmentDetector", model_path: #MISD_MODEL_PATH,
                       apply_sigmoid: true}}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: models: [#MISD_MODEL]
}


#RUN_ENCODE: _ | *false
#RUN_DEFECT: _ | *false
#RUN_BINARIZE_DEFECT: _ | *false
#RUN_MASK_ENCODE: _ | *false
#RUN_ALIGN: _ | *false
#RUN_MISD: _ | *false
#PRECOMPUTED_ENCODINGS: _ | *""
#PRECOMPUTED_MASKED_ENCODINGS: _ | *""
#TMP_DIR_DEFECT: string | *""

#PAIR_FLOW_TMPL: {
    "@type": "build_pairwise_alignment_flow"
    bbox?: _
    bbox_list?: _
    src_image_path: #IMG_PATH
    project_folder?: _
    z_offsets: _ | *[-1, -2]

    run_compute_field: _ | *false
    compute_field_flow_kwargs: #COMPUTE_FIELD_FLOW_SCHEMA & {
        dst_factory_kwargs: {
            info_chunk_size: [2048, 2048, 1]
            per_scale_config: {
                "encoding": "zfpc",
                "zfpc_correlated_dims": [true, true, false, false],
                "zfpc_tolerance": 0.001953125,
            }
        }
        // processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        processing_chunk_sizes: _ | *[[2048, 2048, 1]]
        crop_pad: [64, 64, 0]
    }

    run_invert_field: _ | *false
    invert_field_flow_kwargs: #INVERT_FIELD_FLOW_SCHEMA & {
        dst_factory_kwargs: {
            per_scale_config: {
                "encoding": "zfpc",
                "zfpc_correlated_dims": [true, true, false, false],
                "zfpc_tolerance": 0.001953125,
            }
        }
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad: [64, 64, 0]
        fn: {"@type": "invert_field", "@mode": "partial"}
    }

    run_warp: _ | *false
    warp_flow_kwargs: #WARP_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "imgs_warped"
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad: [256, 256, 0]
        dst_resolution: _ | *[32, 32, #IMG_RES[2]]
    }
    
    run_enc_warped_imgs: _ | *false
    enc_warped_imgs_flow_kwargs: #SUBCHUNKABLE_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "imgs_warped_encoded"
        processing_chunk_sizes: _ | *[[2048, 2048, 1]]
        crop_pad:               [32, 32, 0]
        model: #MISD_ENCODER
        reencode_tgt?: {
            src_path?: string    // defaults to src_image_path
            dst_path?: string    // defaults to "encodings_misd"
        }
    }

    run_misd: _ | *false
    misd_flow_kwargs: #SUBCHUNKABLE_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "misalignments"
        dst_resolution: #MISD_ENCODER.dst_resolution
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad:               [32*6, 32*6, 0]
        models: [...#MISD_MODEL_TMPL]   // one for each z, or one for all
    }
}
#MISD_ENCODER: #ENCODING_MODEL_TMPL

#ENCODING_MODEL_TMPL: {
    path: _
    dst_resolution: _
    dst_path?: _            // can override dst_path
    res_change_mult?: _     // defaults to [1, 1, 1]
    max_processing_chunk_size?: _
    fn: _ | *{
        "@type": "BaseCoarsener"
        model_path: path
        ds_factor: res_change_mult[0]
        tile_size: null     // don't use tiling
    }
    fn_kwargs?: _
    op_kwargs?: _
    subchunkable_kwargs?: _
}

#MISD_MODEL_TMPL: {
    fn: _
}

#MASK_ENCODING_MODEL_TMPL: {
    dst_resolution: _
    fn?: _
    fn_kwargs?: _
    src_path?: _
    src_layer?: _
}
#MASK_ENCODING_MODEL_LIST_TMPL: [...#MASK_ENCODING_MODEL_TMPL]

#WARP_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    dst_path?: string                       // defaults to "warped_imgs"
    dst_resolution:  [int, int, int]        // output warping res
    field_path?: string
    field_resolution?: [int, int, int]      // default: invert_field res
    z_offsets?: [int, int, int]             // default: sets by project
    dst_resolution?: [int, int, int]    // defaults to last res in compute field
}

#INVERT_FIELD_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    dst_path?: string       // defaults to "fields_inv"
    src_path?: string
    z_offsets?: [int, int, int]    // sets by project
    fn: _
    fn_kwargs?: _
    dst_resolution?: [int, int, int]    // defaults to last res in compute field
}

#LAYER_FACTORY_SCHEMA: {
    path?: string
    resolution_list?: [...[int, int, int]]  // list of res to generate/keep
                                            // default behavior varies per flow
    add_zfpc?: _
    ... // any other kwargs for build_cv_layer
    // note: if you need more customization just make and provide a layer
}

#BASIC_FLOW_SCHEMA: {
    dst_layer?: _
    dst_path?: string  // to be used with dst_factory_kwargs if dst_layer is not provided
    dst_factory_kwargs?: #LAYER_FACTORY_SCHEMA
    processing_chunk_sizes: [...[int, int, int]]
    crop_pad?: [int, int, int] | *[0, 0, 0]    // crops for L1 chunks
    ...
}

#SUBCHUNKABLE_FLOW_SCHEMA: #BASIC_FLOW_SCHEMA & {
    subchunkable_kwargs?: _
    op_kwargs?: _
    ...
}

#COMPUTE_FIELD_STAGE_SCHEMA: {
    dst_resolution: [int, int, int]
    fn:         _ | *{"@type": "align_with_online_finetuner", "@mode": "partial"}
    fn_kwargs?: _           // args to `fn`
    path?:      string      // shorthand to override src & tgt
    ...                     // any other kwargs for ComputeFieldStage
}

#COMPUTE_FIELD_FLOW_SCHEMA: #BASIC_FLOW_SCHEMA & {
    src_layer?: _
    src_path?: string   // defaults to "encodings_masked" or "encodings" depending on `skipped_defects`
    tgt_layer?: _
    tgt_path?: string   // defaults is src_path
    stages: [...#COMPUTE_FIELD_STAGE_SCHEMA]
    z_offsets?: [...int]                // sets by project-wide value if empty
    compute_field_multistage_kwargs?: _ // kwargs for `build_compute_field_multistage_flow`
    compute_field_stage_kwargs?: _      // kwargs for `ComputeFieldStage`
    shrink_bbox_to_z_offsets?: _ | *false
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
    num_procs: #GCP_FLOW.num_procs
    semaphores_spec: {
        read: num_procs
        write: num_procs
        cuda: #GCP_FLOW.semaphores_spec.cuda
        cpu: num_procs
    }
    target: _
}
#USE_GPU: #RUN_ENCODE || #RUN_DEFECT || #RUN_ALIGN || #RUN_MISD
#GCP_FLOW: {
    "@type":      "mazepa.execute_on_gcp_with_sqs"
    worker_cluster_region:  "us-east1"
    // worker_cluster_project: "zetta-research"
    // worker_cluster_name:    "zutils-x3"
    // worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:tri-240117-test-ant"
    worker_cluster_project: "zetta-dkronauer-ant-001"
    worker_cluster_name:    "zutils-x2"
    worker_image:           "us-east1-docker.pkg.dev/zetta-dkronauer-ant-001/zutils/zetta_utils:tri-240118-ant-prod-1"
    worker_resources: {
        if #USE_GPU {
            memory: "21000Mi"       // sized for n1
            "nvidia.com/gpu": "1"
        }
        if #USE_GPU == false {
            memory: "30000Mi"       // sized for e2-highmem-4
        }
    }
    worker_replicas:     #CLUSTER_NUM_WORKERS
    local_test:          #TEST_LOCAL
    target: _
    num_procs?: int
    semaphores_spec?: _
}
#TOP_LEVEL_FLOW & {
    target: #PAIR_FLOW_TMPL & {bbox: #BBOX_LIST[0]}
}

#GENERAL_ENC_MODELS_V2: {
    "32": {
        path: "gs://alignment_models/general_encoders_2023/32_32_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096]
    },
    "64": {
        path: "gs://alignment_models/general_encoders_2023/32_64_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048]
    },
    "128": {
        path: "gs://alignment_models/general_encoders_2023/32_128_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024]
    },
    "256": {
        path: "gs://alignment_models/general_encoders_2023/32_256_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512]
    },
    "512": {
        path: "gs://alignment_models/general_encoders_2023/32_512_C1/2023-11-20.static-2.0.1-model.jit"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256]
    },
}

#GENERAL_ENC_MODELS_V1: {
    "32": {
        path: "gs://alignment_models/general_encoders_2023/32_32/last.static-2.0.1-model.jit"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096]
    },
    "64": {
        path: "gs://alignment_models/general_encoders_2023/32_64/last.static-2.0.1-model.jit"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048]
    },
    "128": {
        path: "gs://alignment_models/general_encoders_2023/32_128/last.static-2.0.1-model.jit"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024]
    },
    "256": {
        path: "gs://alignment_models/general_encoders_2023/32_256_C1/last.static-2.0.1-model.jit"
        // path: "gs://alignment_models/general_encoders_2023/32_256_C2/last.static-2.0.1-model.jit"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512]
    },
    "512": {
        path: "gs://alignment_models/general_encoders_2023/32_512_C1/last.static-2.0.1-model.jit"
        // path: "gs://alignment_models/general_encoders_2023/32_512_C2/last.static-2.0.1-model.jit"
        // path: "gs://alignment_models/general_encoders_2023/32_512_C3/last.static-2.0.1-model.jit"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256]
    },
}

#CNS_MODELS: {
    // Adapted from https://github.com/ZettaAI/zetta_utils/blob/nkem/zfish-enc/specs/nico/inference/cns/0_encoding_cns/CNS_encoding_pyramid.cue#L47-L88
    "32": {
        path: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096]
    },
    "64": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M4_conv1_unet3_lr0.0001_equi0.5_post1.6_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048]
    },
    "128": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/3_M3_M5_conv2_unet2_lr0.0001_equi0.5_post1.4_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024]
    },
    "256": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/4_M3_M6_conv3_unet1_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512]
    },
    "512": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M3_M7_conv4_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/epoch=0-step=1584-backup.ckpt.model.spec.json"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256]
    },
    "1024": {
        path: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/5_M3_M8_conv5_lr0.0001_equi0.5_post1.03_fmt0.8_cns_all/last.ckpt.static-1.13.1+cu117-model.jit"
        res_change_mult: [32, 32, 1] // 5 3-8: 32-1024
        dst_resolution: [1024, 1024, #IMG_RES[2]]
        max_processing_chunk_size: [128, 128]
    }
}

#GEN_THICK_MODELS: {
    "32": {
        path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [1, 1, 1] // 0 3-3: 32-32
        dst_resolution: [32, 32, #IMG_RES[2]]
        max_processing_chunk_size: [4096, 4096]
    },
    "64": {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [2, 2, 1] // 1 3-4: 32-64
        dst_resolution: [64, 64, #IMG_RES[2]]
        max_processing_chunk_size: [2048, 2048]
    },
    "128": {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [4, 4, 1] // 2 3-5: 32-128
        dst_resolution: [128, 128, #IMG_RES[2]]
        max_processing_chunk_size: [1024, 1024]
    },
    "256": {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [8, 8, 1] // 3 3-6: 32-256
        dst_resolution: [256, 256, #IMG_RES[2]]
        max_processing_chunk_size: [512, 512]
    },
    "512": {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [16, 16, 1] // 4 3-7: 32-512
        dst_resolution: [512, 512, #IMG_RES[2]]
        max_processing_chunk_size: [256, 256]
    }
}
