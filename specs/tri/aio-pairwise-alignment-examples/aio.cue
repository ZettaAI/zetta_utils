#IMG_PATH: "precomputed://gs://dkronauer-ant-001-raw/brain"
#IMG_RES: [4, 4, 42]
#IMG_SIZE: [102400, 96256, 6112]

#TEST_SMALL: true
#TEST_LOCAL: true
#CLUSTER_NUM_WORKERS: 16

#RUN_ENCODE: true
#RUN_DEFECT: true
#RUN_MASK_ENCODE: true
#RUN_ALIGN: true
#RUN_MISD: true

#PROJECT: "test-240222"

#PAIR_FLOW_TMPL: project_folder: _ | *"gs://dkronauer-ant-001-alignment/\(#PROJECT)"

// #PRECOMPUTED_ENCODINGS: "gs://dkronauer-ant-001-alignment/test-240109-z50-gen-v3-32nm-1/encodings"
// #PRECOMPUTED_MASKED_ENCODINGS: "gs://dkronauer-ant-001-alignment/test-240109-z50-gen-v3-64nm-v2-defect-opening-2/masked_encodings"

//////////////////////////////////
// DEFECT DETECTOR CONFIG       //
//////////////////////////////////
#DEFECT_MODEL_PATH: "gs://zetta_lee_fly_cns_001_models/jit/20221114-defects-step50000.static-1.11.0.jit"
#DEFECT_MODEL_RES: [64, 64, #IMG_RES[2]]
#PAIR_FLOW_TMPL: defect_flow_kwargs: fn: model_path: #DEFECT_MODEL_PATH

//////////////////////////////////
// MISALIGNMENT DETECTOR CONFIG //
//////////////////////////////////
if true {
    #MISD_MODEL_PATH: "gs://zetta-research-nico/training_artifacts/aced_misd_general/3.2.0_dsfactor2_thr1.5_lr0.0001_z1/epoch=44-step=11001-backup.ckpt.model.spec.json"
    #MISD_ENCODER: #GENERAL_ENC_MODELS_V2["64"]
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: dst_path: "imgs_warped_encoded-v2-z1"
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: reencode_tgt: {dst_path: "encodings_misd-v2-z1"}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: dst_path: "misalignments-v2-z1"
    #MISD_MODEL: {fn: {"@type": "MisalignmentDetector", model_path: #MISD_MODEL_PATH,
                       apply_sigmoid: true}}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: models: [#MISD_MODEL]
}

if #TEST_SMALL {
    #BBOX: {#BBOX_TMPL & {start_coord: [11*4096,  5*4096,  50]
                          end_coord:   [18*4096, 10*4096, 104]}}
    #PAIR_FLOW_TMPL: z_offsets: [-1, -2]

    // Don't align bogus pairs (e.g., z=50 to z=49)
    #PAIR_FLOW_TMPL: compute_field_flow_kwargs: shrink_bbox_to_z_offsets: true
    #PAIR_FLOW_TMPL: invert_field_flow_kwargs: shrink_bbox_to_z_offsets: true
    #PAIR_FLOW_TMPL: warp_flow_kwargs: shrink_bbox_to_z_offsets: true
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: shrink_bbox_to_z_offsets: true
    #PAIR_FLOW_TMPL: misd_flow_kwargs: shrink_bbox_to_z_offsets: true
    #ENCODING_MODELS: [...#ENCODING_MODEL_TMPL]
    #PAIR_FLOW_TMPL: encoding_flow_kwargs: models: #ENCODING_MODELS
    #ENCODING_MODELS: [
        #GEN_THICK_MODELS["512"],
        #GEN_THICK_MODELS["256"],
        #GEN_THICK_MODELS["128"],
        #GEN_THICK_MODELS["64"],
        #GEN_THICK_MODELS["32"],
    ]
    #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: dst_resolution_list: [
        {dst_resolution: [512, 512, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [256, 256, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [128, 128, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [ 64,  64, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [ 32,  32, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
    ]
    #PAIR_FLOW_TMPL: compute_field_flow_kwargs: stages: [
        {dst_resolution: [512, 512, #IMG_RES[2]], fn_kwargs: {sm: 300, num_iter: 700, lr: 0.015}},
        {dst_resolution: [256, 256, #IMG_RES[2]], fn_kwargs: {sm: 150, num_iter: 700, lr: 0.030}},
        {dst_resolution: [128, 128, #IMG_RES[2]], fn_kwargs: {sm: 100, num_iter: 500, lr: 0.050}},
        {dst_resolution: [ 64,  64, #IMG_RES[2]], fn_kwargs: {sm:  50, num_iter: 300, lr: 0.100}},
        {dst_resolution: [ 32,  32, #IMG_RES[2]], fn_kwargs: {sm:  25, num_iter: 200, lr: 0.100}},
    ]

    #PAIR_FLOW_TMPL: run_encoding: #RUN_ENCODE
    if #RUN_ENCODE == false
        if #PRECOMPUTED_ENCODINGS != "" {
            #PAIR_FLOW_TMPL: encoding_flow_kwargs: dst_path: #PRECOMPUTED_ENCODINGS
        }
    #PAIR_FLOW_TMPL: run_defect: #RUN_DEFECT
    #PAIR_FLOW_TMPL: run_binarize_defect: #RUN_DEFECT
    if #RUN_MASK_ENCODE == true {
        #PAIR_FLOW_TMPL: run_mask_encodings: true
    }
    if #RUN_MASK_ENCODE == false {
        if #PRECOMPUTED_MASKED_ENCODINGS != "" {
            #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: dst_path: #PRECOMPUTED_MASKED_ENCODINGS
        }
    }
    if #RUN_ALIGN {
        #PAIR_FLOW_TMPL: run_compute_field: true
        #PAIR_FLOW_TMPL: run_invert_field: true
        #PAIR_FLOW_TMPL: run_warp: true
    }
    if #RUN_MISD {
        #PAIR_FLOW_TMPL: run_enc_warped_imgs: true
        #PAIR_FLOW_TMPL: run_misd: true
    }

    ///////////////////////////////
    // HACKS FOR RUNNING CUTOUTS //
    ///////////////////////////////

    // Reduce processing chunk size for cutouts
    #PAIR_FLOW_TMPL: encoding_flow_kwargs: processing_chunk_sizes: [[512, 512, 1]]
    #PAIR_FLOW_TMPL: defect_flow_kwargs: processing_chunk_sizes: [[512, 512, 1]]
    #PAIR_FLOW_TMPL: binarize_defect_flow_kwargs: processing_chunk_sizes: [[1024, 1024, 1]]
    #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: processing_chunk_sizes: [[1024, 1024, 1]]
    #PAIR_FLOW_TMPL: compute_field_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    #PAIR_FLOW_TMPL: invert_field_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    #PAIR_FLOW_TMPL: warp_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    #PAIR_FLOW_TMPL: misd_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]

    // Use intermediaries to avoid unaligned writes for testing cutouts
    let use_intermediaries = {skip_intermediaries: false}
    #TMP_DIRS: [...{...}]
    if #TEST_LOCAL {
        #TMP_DIRS: [
            {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp0/"]},
            {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp1/"]},
            {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp2/"]},
            {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp3/"]},
        ]
    }
    if #TEST_LOCAL == false {
        #TMP_DIRS: [
            {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp0/"]},
            {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp1/"]},
            {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp2/"]},
            {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp3/"]},
        ]
    }
    #PAIR_FLOW_TMPL: encoding_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[0]
    #PAIR_FLOW_TMPL: defect_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[1]
    #PAIR_FLOW_TMPL: binarize_defect_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[2]
    #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[3]

    // Avoid unaligned writes by adjusting dst offsets
    let output_voxel_offset = [#BBOX.start_coord[0], #BBOX.start_coord[1], 0]
    let info_voxel_offset_map_ = {
        "1024_1024_\(#IMG_RES[2])": [output_voxel_offset[0]/(1024/4), output_voxel_offset[1]/(1024/4), 0]
        "512_512_\(#IMG_RES[2])": [output_voxel_offset[0]/(512/4), output_voxel_offset[1]/(512/4), 0]
        "256_256_\(#IMG_RES[2])": [output_voxel_offset[0]/(256/4), output_voxel_offset[1]/(256/4), 0]
        "128_128_\(#IMG_RES[2])": [output_voxel_offset[0]/(128/4), output_voxel_offset[1]/(128/4), 0]
        "64_64_\(#IMG_RES[2])": [output_voxel_offset[0]/(64/4), output_voxel_offset[1]/(64/4), 0]
        "32_32_\(#IMG_RES[2])": [output_voxel_offset[0]/(32/4), output_voxel_offset[1]/(32/4), 0]
    }
    #PAIR_FLOW_TMPL: compute_field_flow_kwargs: dst_factory_kwargs: info_voxel_offset_map: info_voxel_offset_map_
    #PAIR_FLOW_TMPL: warp_flow_kwargs: dst_factory_kwargs: info_voxel_offset_map: info_voxel_offset_map_
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: dst_factory_kwargs: info_voxel_offset_map: info_voxel_offset_map_
    #BBOX_LIST: [#BBOX]
}


#RUN_ENCODE: _ | *false
#RUN_DEFECT: _ | *false
#RUN_MASK_ENCODE: _ | *false
#RUN_ALIGN: _ | *false
#RUN_MISD: _ | *false
#PRECOMPUTED_ENCODINGS: _ | *""
#PRECOMPUTED_MASKED_ENCODINGS: _ | *""

#PAIR_FLOW_TMPL: {
    "@type": "build_pairwise_alignment_flow"
    bbox?: _
    bbox_list?: _
    src_image_path: #IMG_PATH
    project_folder?: _
    z_offsets: _ | *[-1, -2]

    run_encoding: _ | *false
    encoding_flow_kwargs: #ENCODING_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "encodings"
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [4096, 4096, 1]]
        crop_pad:               _ | *[32, 32, 0]
        models: _
    }

    run_defect: _ | *false
    defect_flow_kwargs: #SUBCHUNKABLE_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "defect"
        dst_resolution: #DEFECT_MODEL_RES
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [512, 512, 1]]
        crop_pad:               _ | *[512, 512, 0]  // good for processing_chunk_size=512
        fn: _ | *{
            "@type": "DefectDetector"
            model_path: _
            ds_factor?: _
            tile_size: null     // don't use tiling
        }
    }

    run_binarize_defect: _ | *false
    binarize_defect_flow_kwargs: #SUBCHUNKABLE_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "defect_binarized"
        dst_resolution: #DEFECT_MODEL_RES
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [1024, 1024, 1]]
        crop_pad:               _ | *[128, 128, 0]
        fn: _ | *{"@type": "binarize_defect_prediction", "@mode": "partial"}
        fn_kwargs: _ | *{
            threshold: 100
            kornia_opening_width: 11
            kornia_dilation_width: 3
            // filter_cc_threshold: 320
            filter_cc_threshold: 240
            // kornia_closing_width: 25
            kornia_closing_width: 30
        }
    }

    run_mask_encodings: _ | *false
    mask_encodings_flow_kwargs: #MASK_ENCODINGS_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "encodings_masked"
        fn: _ | *{"@type": "zero_out_src_with_mask", "@mode": "partial"}
        mask_resolution: #DEFECT_MODEL_RES
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [1024, 1024, 1]]
    }

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
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
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
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad: [256, 256, 0]
        dst_resolution: _ | *[32, 32, #IMG_RES[2]]
    }

    run_enc_warped_imgs: _ | *false
    enc_warped_imgs_flow_kwargs: #ENCODE_WARPED_IMGS_FLOW_SCHEMA & {
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [4096, 4096, 1]]
        crop_pad:               [32, 32, 0]
        model: #MISD_ENCODER
    }

    run_misd: _ | *false
    misd_flow_kwargs: #MISALIGNMENT_DETECTOR_FLOW_SCHEMA & {
        dst_resolution: #MISD_ENCODER.dst_resolution
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad:               [32*6, 32*6, 0]
        models: [...#MISD_MODEL_TMPL]   // one for each z, or one for all
    }
}

#DEFAULT_LAYER_FACTORY_SCHEMA: {
    path?: string
    resolution_list?: [...[int, int, int]]  // list of res to generate/keep
    per_scale_config?: _                    // dict of attrs to be added to each scale
    ...                                     // any other kwargs for build_cv_layer
}

#COMMON_FLOW_SCHEMA: {
    dst_layer?: _
    dst_path?: string       // To be used with dst_factory_kwargs if dst_layer is not provided
    dst_factory?: _         // Should be a Callable(), defaults to DEFAULT_LAYER_FACTORY_SCHEMA
    dst_factory_kwargs?: _
    ...
}

#SUBCHUNKABLE_FLOW_SCHEMA: #COMMON_FLOW_SCHEMA & {
    dst_resolution?: [int, int, int]
    op?: _
    op_kwargs?: _       // kwargs for `op` (not build_subchunkable_apply_flow's op_kwargs)
    fn?: _
    fn_kwargs?: _
    processing_chunk_sizes: [...[int, int, int]]
    crop_pad?: [int, int, int]      // crops for L1 chunks
    subchunkable_kwargs?: _         // kwargs for `build_subchunkable_apply_flow`
    ...
}

#ENCODING_MODEL_TMPL: {
    path: _                                 // model path
    dst_resolution: _                       // output res of this encoder
    dst_path?: _                            // override flow's dst_path
    res_change_mult?: [int, int, int]       // defaults to [1, 1, 1]
    max_processing_chunk_size?: [int, int]  // restricts processing_chunk_size[0:1]
    fn: _ | *{
        "@type": "BaseCoarsener"
        model_path: path
        ds_factor: res_change_mult[0]
        tile_size: null     // don't use tiling
    }
    fn_kwargs?: _               // per-model overrides
    op_kwargs?: _               // per-model overrides
    subchunkable_kwargs?: _     // per-model overrides
}

#ENCODING_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    models: [...#ENCODING_MODEL_TMPL]
}

#MASK_FN_TMPL: {
    dst_resolution: _           // output res of this masking step
    fn_kwargs?: _               // per-model overrides
    src_path?: _
    src_layer?: _
}

#MASK_ENCODINGS_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    dst_resolution_list: [...#MASK_FN_TMPL]
    mask_path?: string
    mask_layer?: _
    mask_resolution: [int, int, int]
}

#COMPUTE_FIELD_STAGE_SCHEMA: {
    dst_resolution: [int, int, int]
    fn:         _ | *{"@type": "align_with_online_finetuner", "@mode": "partial"}
    fn_kwargs?: _           // args for `fn`
    path?:      string      // shorthand to override src & tgt
    ...                     // any other kwargs for `ComputeFieldStage`
}

#COMPUTE_FIELD_FLOW_SCHEMA: #COMMON_FLOW_SCHEMA & {
    src_layer?: _
    src_path?: string
    tgt_layer?: _
    tgt_path?: string                           // defaults to src_path
    stages: [...#COMPUTE_FIELD_STAGE_SCHEMA]
    z_offsets?: [...int]                        // sets by project-wide value if empty
    compute_field_multistage_kwargs?: _         // kwargs for `build_compute_field_multistage_flow`
    compute_field_stage_kwargs?: _              // kwargs for `ComputeFieldStage`
    shrink_bbox_to_z_offsets?: _ | *false
}

#INVERT_FIELD_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    dst_path?: string                   // defaults to "fields_inv"
    src_path?: string
    z_offsets?: [...int]                // sets by project
    dst_resolution?: [int, int, int]    // defaults to last res in compute field
    shrink_bbox_to_z_offsets?: bool
}

#WARP_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    dst_path?: string                       // defaults to "imgs_warped"
    dst_resolution:  [int, int, int]        // output warping res
    field_path?: string
    field_resolution?: [int, int, int]      // defaults to invert_field's res
    z_offsets?: [int, int, int]             // defaults to project's
    dst_resolution?: [int, int, int]        // defaults to last res in compute field
    shrink_bbox_to_z_offsets?: bool
}

#REENCODE_TGT_OPTIONS: {
    src_path?: _    // input path, defaults to flow's
    dst_path: _     // output path
}

#MISD_ENCODER: #ENCODING_MODEL_TMPL

#ENCODE_WARPED_IMGS_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    model: #ENCODING_MODEL_TMPL
    dst_path?: string                       // defaults to "imgs_warped_encoded"
    z_offsets?: [...int]                    // defaults to project's
    dst_resolution?: [int, int, int]        // defaults to last res in compute field
    shrink_bbox_to_z_offsets?: bool
    reencode_tgt?: #REENCODE_TGT_OPTIONS    // if misd's encoder is different from alignment's encoders
}

#MISD_MODEL_TMPL: {
    dst_resolution?: [int, int, int]  // defaults to flow's
    fn: _
    max_processing_chunk_size?: [int, int]
}

#MISALIGNMENT_DETECTOR_FLOW_SCHEMA: #SUBCHUNKABLE_FLOW_SCHEMA & {
    models: [...#MISD_MODEL_TMPL]   // One per z_offset. Duplicate if only 1 is given.
    tgt_layer?: _
    tgt_path?: string                           // defaults to src_path
    shrink_bbox_to_z_offsets?: bool
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
    worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:tri-240117-test-ant"
    worker_resources: {
        memory: "21000Mi"       // sized for n1-highmem-4
        "nvidia.com/gpu": "1"
    }
    worker_replicas:     #CLUSTER_NUM_WORKERS
    local_test:          #TEST_LOCAL
    target: _
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
