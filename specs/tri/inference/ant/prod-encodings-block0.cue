#IMG_PATH: "precomputed://gs://dkronauer-ant-001-raw/brain"
#IMG_RES: [4, 4, 42]
#IMG_SIZE: [102400, 96256, 6112]

// #TEST_LOCAL: true
// #CLUSTER_NUM_WORKERS: 16
#CLUSTER_NUM_WORKERS: 48

#RUN_ENCODE: true
#RUN_DEFECT: true
// #RUN_BINARIZE_DEFECT: true
// #RUN_MASK_ENCODE: true
// #RUN_ALIGN: true
// #RUN_MISD: true

#PROJECT: "production-240118"
#PAIR_FLOW_TMPL: project_folder: _ | *"gs://dkronauer-ant-001-alignment/\(#PROJECT)"

#OUTPUT_PATH_ENCODINGS:         "gs://dkronauer-ant-001-encodings/\(#PROJECT)/encodings"
#OUTPUT_PATH_DEFECT:            "gs://dkronauer-ant-001-encodings/\(#PROJECT)/defect"
#OUTPUT_PATH_BINARIZED_DEFECT:  "gs://dkronauer-ant-001-encodings/\(#PROJECT)/defect_binarized"
#OUTPUT_PATH_MASKED_ENCODINGS:  "gs://dkronauer-ant-001-encodings/\(#PROJECT)/encodings_masked"
#PAIR_FLOW_TMPL: encoding_flow_kwargs:          dst_path: #OUTPUT_PATH_ENCODINGS
#PAIR_FLOW_TMPL: defect_flow_kwargs:            dst_path: #OUTPUT_PATH_DEFECT
#PAIR_FLOW_TMPL: binarize_defect_flow_kwargs:   dst_path: #OUTPUT_PATH_BINARIZED_DEFECT
#PAIR_FLOW_TMPL: mask_encodings_flow_kwargs:    dst_path: #OUTPUT_PATH_MASKED_ENCODINGS

#GCP_FLOW: num_procs: 4
#GCP_FLOW: semaphores_spec: {
    read: #GCP_FLOW.num_procs
    write: #GCP_FLOW.num_procs
    cuda: 1
    cpu: #GCP_FLOW.num_procs
}

// #PRECOMPUTED_ENCODINGS: "gs://dkronauer-ant-001-alignment/test-240109-z50-gen-v3-32nm-1/encodings"
// #PRECOMPUTED_MASKED_ENCODINGS: "gs://dkronauer-ant-001-alignment/test-240109-z50-gen-v3-32nm-1/masked_encodings"
// #PRECOMPUTED_MASKED_ENCODINGS: "gs://dkronauer-ant-001-alignment/test-240109-z50-gen-v3-64nm-v2-defect-opening-2/masked_encodings"

//////////////////////////////////
// DEFECT DETECTOR CONFIG       //
//////////////////////////////////
// #DEFECT_MODEL_PATH: "gs://zetta_lee_fly_cns_001_models/jit/20221114-defects-step50000.static-1.11.0.jit"
#DEFECT_MODEL_PATH: "gs://dkronauer-ant-001-binaries/20221114-defects-step50000.static-1.11.0.jit"
#DEFECT_MODEL_RES: [64, 64, #IMG_RES[2]]
#PAIR_FLOW_TMPL: defect_flow_kwargs: fn: model_path: #DEFECT_MODEL_PATH
#PAIR_FLOW_TMPL: defect_flow_kwargs: {
    // This model performs best at 512x512 blocks
    processing_chunk_sizes: [[2048, 2048, 1], [1024, 1024, 1], [512, 512, 1]]
    crop_pad: [512, 512, 0]
}


//////////////////////////////////
// MISALIGNMENT DETECTOR CONFIG //
//////////////////////////////////
if true {
// if false {
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
// if true {
if false {
    // configs for misd-v2, z-2 model
    #MISD_MODEL_PATH: "gs://zetta-research-nico/training_artifacts/aced_misd_general/3.2.0_dsfactor2_thr2.0_lr0.0001_z2/epoch=74-step=18730-backup.ckpt.model.spec.json"
    #MISD_ENCODER: #GENERAL_ENC_MODELS_V2["64"]
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: dst_path: "imgs_warped_encoded-v2-z2"
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: reencode_tgt: {dst_path: "encodings_misd-v2-z2"}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: dst_path: "misalignments-v2-z2"
    #MISD_MODEL: {fn: {"@type": "MisalignmentDetector", model_path: #MISD_MODEL_PATH,
                       apply_sigmoid: true}}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: models: [#MISD_MODEL]
}
// if true {
if false {
    // configs for misd-cns
    #MISD_MODEL_PATH: "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr5.0_lr0.00001_z1z2_400-500_2910-2920_more_aligned_unet5_32_finetune_2/last.ckpt.static-2.0.0+cu117-model.jit"
    #MISD_ENCODER: #CNS_MODELS["32"]
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: dst_path: "imgs_warped_encoded_cns"
    #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: reencode_tgt: {dst_path: "encodings_misd-cns"}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: dst_path: "misalignments-cns"
    #MISD_MODEL: {fn: {"@type": "MisalignmentDetector", model_path: #MISD_MODEL_PATH,
                       apply_sigmoid: false}}
    #PAIR_FLOW_TMPL: misd_flow_kwargs: models: [#MISD_MODEL]
}


if true {
    // #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0,   0]
                          // end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 152]}}
    // #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0, 153]
                          // end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 874]}}
    #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0, 152]
                          end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 153]}}
    // #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0, 874]
    //                       end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 6112]}}
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

    #PAIR_FLOW_TMPL: run_encoding: #RUN_ENCODE
    #PAIR_FLOW_TMPL: run_defect: #RUN_DEFECT
    #PAIR_FLOW_TMPL: run_binarize_defect: #RUN_BINARIZE_DEFECT
    #PAIR_FLOW_TMPL: run_mask_encodings: #RUN_MASK_ENCODE

    ///////////////////////////////
    //   HACKS FOR PERFORMANCE   //
    ///////////////////////////////
    // Need to set lv1 punk size manually to be 2x2 of l0 for each model
    // because of a bug that makes >2x2 lv1 size unrunnable
    #GEN_THICK_MODELS: "512": subchunkable_kwargs: {
        processing_chunk_sizes: [[512, 512, 1], [256, 256, 1]]
        processing_crop_pads:   [[0, 0, 0], [32, 32, 0]]
    }
    #GEN_THICK_MODELS: "256": subchunkable_kwargs: {
        processing_chunk_sizes: [[1024, 1024, 1], [512, 512, 1]]
        processing_crop_pads:   [[0, 0, 0], [32, 32, 0]]
    }
    #GEN_THICK_MODELS: "128": subchunkable_kwargs: {
        processing_chunk_sizes: [[2048, 2048, 1], [1024, 1024, 1]]
        processing_crop_pads:   [[0, 0, 0], [32, 32, 0]]
    }
    #GEN_THICK_MODELS: "64": subchunkable_kwargs: {
        processing_chunk_sizes: [[4096, 4096, 1], [2048, 2048, 1]]
        processing_crop_pads:   [[0, 0, 0], [32, 32, 0]]
    }
    #GEN_THICK_MODELS: "32": subchunkable_kwargs: {
        processing_chunk_sizes: [[4096, 4096, 1], [2048, 2048, 1]]
        processing_crop_pads:   [[0, 0, 0], [32, 32, 0]]
        // processing_chunk_sizes: [[3*1024, 3*1024, 1]]
        // processing_crop_pads:   [[32, 32, 0]]
    }
    #PAIR_FLOW_TMPL: encoding_flow_kwargs: dst_factory_kwargs: {
        // Match lv0 punk size to avoid extra writes to intermediaries
        // TODO: debug why remote local intermediaries are bugged
        info_chunk_size_map: {
            "32_32_\(#IMG_RES[2])": [2048, 2048, 1]
            "64_64_\(#IMG_RES[2])": [2048, 2048, 1]
            "128_128_\(#IMG_RES[2])": [1024, 1024, 1]
            "256_256_\(#IMG_RES[2])": [512, 512, 1]
            "512_512_\(#IMG_RES[2])": [256, 256, 1]
        }
    }

    // #TMP_DIR_ENCODINGS: #TMP_DIRS[0].level_intermediaries_dirs[0]
    #TMP_DIR_DEFECT: #TMP_DIRS[1].level_intermediaries_dirs[0]

    ///////////////////////////////
    // HACKS FOR RUNNING CUTOUTS //
    ///////////////////////////////

    // // Reduce processing chunk size for cutouts
    // #PAIR_FLOW_TMPL: encoding_flow_kwargs: processing_chunk_sizes: [[512, 512, 1]]
    // #PAIR_FLOW_TMPL: defect_flow_kwargs: processing_chunk_sizes: [[512, 512, 1]]
    // #PAIR_FLOW_TMPL: binarize_defect_flow_kwargs: processing_chunk_sizes: [[1024, 1024, 1]]
    // #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: processing_chunk_sizes: [[1024, 1024, 1]]
    // #PAIR_FLOW_TMPL: compute_field_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    // #PAIR_FLOW_TMPL: invert_field_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    // #PAIR_FLOW_TMPL: warp_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    // #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]
    // #PAIR_FLOW_TMPL: misd_flow_kwargs: processing_chunk_sizes: [[2048, 2048, 1]]

    // Use intermediaries to avoid unaligned writes of encoders
    // let use_intermediaries = {skip_intermediaries: false}
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
            {level_intermediaries_dirs: ["gs://dkronauer-ant-001-alignment-tmp/\(#PROJECT)/tmp0/"]},
            {level_intermediaries_dirs: ["gs://dkronauer-ant-001-alignment-tmp/\(#PROJECT)/tmp1/"]},
            {level_intermediaries_dirs: ["gs://dkronauer-ant-001-alignment-tmp/\(#PROJECT)/tmp2/"]},
            {level_intermediaries_dirs: ["gs://dkronauer-ant-001-alignment-tmp/\(#PROJECT)/tmp3/"]},
            // {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp0/"]},
            // {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp1/"]},
            // {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp2/"]},
            // {level_intermediaries_dirs: ["gs://tmp_2w/ant/\(#PROJECT)/tmp3/"]},
            // {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp0/"]},
            // {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp1/"]},
            // {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp2/"]},
            // {level_intermediaries_dirs: ["file://~/.zetta_utils/tmp3/"]},
        ]
    }
    // #PAIR_FLOW_TMPL: encoding_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[0]
    // #PAIR_FLOW_TMPL: defect_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[1]
    // #PAIR_FLOW_TMPL: binarize_defect_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[2]
    // #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: subchunkable_kwargs: use_intermediaries & #TMP_DIRS[3]

    // // Avoid unaligned writes by adjusting dst offsets
    // let output_voxel_offset = [#BBOX.start_coord[0], #BBOX.start_coord[1], 0]
    // let info_voxel_offset_map_ = {
    //     "1024_1024_\(#IMG_RES[2])": [output_voxel_offset[0]/(1024/4), output_voxel_offset[1]/(1024/4), 0]
    //     "512_512_\(#IMG_RES[2])": [output_voxel_offset[0]/(512/4), output_voxel_offset[1]/(512/4), 0]
    //     "256_256_\(#IMG_RES[2])": [output_voxel_offset[0]/(256/4), output_voxel_offset[1]/(256/4), 0]
    //     "128_128_\(#IMG_RES[2])": [output_voxel_offset[0]/(128/4), output_voxel_offset[1]/(128/4), 0]
    //     "64_64_\(#IMG_RES[2])": [output_voxel_offset[0]/(64/4), output_voxel_offset[1]/(64/4), 0]
    //     "32_32_\(#IMG_RES[2])": [output_voxel_offset[0]/(32/4), output_voxel_offset[1]/(32/4), 0]
    // }
    // #PAIR_FLOW_TMPL: compute_field_flow_kwargs: dst_factory_kwargs: info_voxel_offset_map: info_voxel_offset_map_
    // #PAIR_FLOW_TMPL: warp_flow_kwargs: dst_factory_kwargs: info_voxel_offset_map: info_voxel_offset_map_
    // #PAIR_FLOW_TMPL: enc_warped_imgs_flow_kwargs: dst_factory_kwargs: info_voxel_offset_map: info_voxel_offset_map_
    #BBOX_LIST: [#BBOX]
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

    run_encoding: _ | *false
    encoding_flow_kwargs: {
        dst_path?: _  // defaults to "encodings"
        // processing_chunk_sizes: _ | *[[4096, 4096, 1], [4096, 4096, 1]]
        processing_chunk_sizes: _ | *[[4096, 4096, 1]]
        crop_pad:               _ | *[32, 32, 0]
        models: _
    }

    run_defect: _ | *false
    defect_flow_kwargs: {
        dst_path?: _  // defaults to "defect"
        dst_resolution: #DEFECT_MODEL_RES
        processing_chunk_sizes: _
        crop_pad?:              _
        subchunkable_kwargs: {
            skip_intermediaries: false
            level_intermediaries_dirs: [#TMP_DIR_DEFECT, #TMP_DIR_DEFECT, #TMP_DIR_DEFECT]
        }
        fn: _ | *{
            "@type": "DefectDetector"
            model_path: _
            ds_factor?: _
            tile_size: null     // don't use tiling
        }
        dst_factory_kwargs: {
            info_chunk_size: [2048, 2048, 1]
        }
    }

    run_binarize_defect: _ | *false
    binarize_defect_flow_kwargs: {
        dst_path?: _  // defaults to "defect_binarized"
        dst_resolution: #DEFECT_MODEL_RES
        processing_chunk_sizes: _ | *[[2048, 2048, 1], [1024, 1024, 1]]
        crop_pad:               _ | *[128, 128, 0]
        fn: _ | *{"@type": "binarize_defect_prediction", "@mode": "partial"}
        fn_kwargs: _ | *{
            threshold: 100
            kornia_opening_width: 11
            kornia_dilation_width: 3
            // filter_cc_threshold: 320
            filter_cc_threshold: 240    // filter less small objects
            // kornia_closing_width: 25
            kornia_closing_width: 30    // connect more disconnected lines
        }
        dst_factory_kwargs: {
            info_chunk_size: [1024, 1024, 1]
        }
    }

    run_mask_encodings: _ | *false
    mask_encodings_flow_kwargs: {
        dst_path?: _  // defaults to "encodings_masked"
        fn: _ | *{"@type": "zero_out_src_with_mask2", "@mode": "partial"}
        dst_resolution_list: [...#MASK_ENCODING_MODEL_TMPL]
        mask_resolution: #DEFECT_MODEL_RES
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad:               _ | *[128, 128, 0]
        fn_kwargs?: _
        dst_factory_kwargs: {
            info_chunk_size: [2048, 2048, 1]
        }
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
        dst_path?: _  // defaults to "imgs_warped"
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [2048, 2048, 1]]
        crop_pad: [256, 256, 0]
        dst_resolution: _ | *[32, 32, #IMG_RES[2]]
    }

    run_enc_warped_imgs: _ | *false
    enc_warped_imgs_flow_kwargs: #SUBCHUNKABLE_FLOW_SCHEMA & {
        dst_path?: _  // defaults to "imgs_warped_encoded"
        processing_chunk_sizes: _ | *[[4096, 4096, 1], [4096, 4096, 1]]
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
    if #USE_GPU {
        worker_resources: {
            memory: "21000Mi"       // sized for n1-highmem-4
            "nvidia.com/gpu": "1"
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

// #GCP_FLOW: {
//     worker_resources: {
//         memory: "21000Mi"       // sized for n1-highmem-4
//         "nvidia.com/gpu": "1"
//     }
// }
// // if #RUN_ENCODE or #RUN_ENCODE or #RUN_ENCODE or #RUN_ENCODE {
// if #RUN_ENCODE  {
//     #GCP_FLOW: worker_resources: {
//         memory: "21000Mi"       // sized for n1-highmem-4
//         "nvidia.com/gpu": "1"
//     }
// }
// if true {
//     #GCP_FLOW: worker_resources: {
//         memory: "21000Mi"       // sized for n1-highmem-4
//         "nvidia.com/gpu": "1"
//     }
// }

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

    //// V1 configs
    // #ENCODING_MODELS: [
        // #GENERAL_ENC_MODELS_V1["512"],
        // #CNS_MODELS["256"],
        // #CNS_MODELS["128"],
        // #GENERAL_ENC_MODELS_V1["64"],
        // #GENERAL_ENC_MODELS_V1["32"],
    // ]
    // #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: dst_resolution_list: [
    //     {dst_resolution: [512, 512, #IMG_RES[2]], fn_kwargs: {grow_mask_width: 3}},
    //     {dst_resolution: [256, 256, #IMG_RES[2]], fn_kwargs: {grow_mask_width: 5}},
    //     {dst_resolution: [128, 128, #IMG_RES[2]], fn_kwargs: {grow_mask_width: 5}},
    //     {dst_resolution: [ 64,  64, #IMG_RES[2]], fn_kwargs: {grow_mask_width: 3}},
    //     {dst_resolution: [ 32,  32, #IMG_RES[2]], fn_kwargs: {grow_mask_width: 3}},
    // ]
    // #PAIR_FLOW_TMPL: compute_field_flow_kwargs: stages: [
    //     {dst_resolution: [512, 512, #IMG_RES[2]], fn_kwargs: {sm: 150, num_iter: 700, lr: 0.015}},
    //     {dst_resolution: [256, 256, #IMG_RES[2]], fn_kwargs: {sm: 100, num_iter: 700, lr: 0.030}},
    //     {dst_resolution: [128, 128, #IMG_RES[2]], fn_kwargs: {sm:  75, num_iter: 500, lr: 0.050}},
    //     {dst_resolution: [ 64,  64, #IMG_RES[2]], fn_kwargs: {sm:  50, num_iter: 300, lr: 0.100}},
    //     {dst_resolution: [ 32,  32, #IMG_RES[2]], fn_kwargs: {sm:  25, num_iter: 200, lr: 0.100}},
    // ]