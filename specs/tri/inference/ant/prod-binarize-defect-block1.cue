#IMG_PATH: "precomputed://gs://dkronauer-ant-001-raw/brain"
#IMG_RES: [4, 4, 42]
#IMG_SIZE: [102400, 96256, 6112]

// #TEST_LOCAL: true
#CLUSTER_NUM_WORKERS: 64
// #CLUSTER_NUM_WORKERS: 48

#RUN_BINARIZE_DEFECT: true
#RUN_MASK_ENCODE: true

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

// defect binarize is very memory intensive so we reduce concurrent cpus
#GCP_FLOW: num_procs: 4
#GCP_FLOW: semaphores_spec: {
    read: #GCP_FLOW.num_procs
    write: #GCP_FLOW.num_procs
    cuda: 1
    cpu: 2
}

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

if true {
    // #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0,   0]
                          // end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 152]}}
    #BBOX: {#BBOX_TMPL & {start_coord: [           0,            0, 152]
                          end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 1300]}}
                          // end_coord:   [#IMG_SIZE[0], #IMG_SIZE[1], 6112]}}
    #BBOX_LIST: [#BBOX]

    #PAIR_FLOW_TMPL: mask_encodings_flow_kwargs: dst_resolution_list: [
        {dst_resolution: [512, 512, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [256, 256, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [128, 128, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [ 64,  64, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
        {dst_resolution: [ 32,  32, #IMG_RES[2]], fn_kwargs: {opening_width: 2, dilation_width: 2}},
    ]

    #PAIR_FLOW_TMPL: run_binarize_defect: #RUN_BINARIZE_DEFECT
    #PAIR_FLOW_TMPL: run_mask_encodings: #RUN_MASK_ENCODE
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
}

#MASK_ENCODING_MODEL_TMPL: {
    dst_resolution: _
    fn?: _
    fn_kwargs?: _
    src_path?: _
    src_layer?: _
}
#MASK_ENCODING_MODEL_LIST_TMPL: [...#MASK_ENCODING_MODEL_TMPL]

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
    worker_cluster_project: "zetta-dkronauer-ant-001"
    worker_cluster_name:    "zutils-x2"
    worker_image:           "us-east1-docker.pkg.dev/zetta-dkronauer-ant-001/zutils/zetta_utils:tri-240118-ant-prod-1"
    worker_resources: {
        if #USE_GPU {
            memory: "21000Mi"       // sized for n1
            "nvidia.com/gpu": "1"
        }
        if #USE_GPU == false {
            memory: "25000Mi"       // sized for e2-highmem-4
        }
    }
    worker_replicas:     #CLUSTER_NUM_WORKERS
    local_test:          #TEST_LOCAL
    target: _
    num_procs?: int
    semaphores_spec?: _
    // do_dryrun_estimation: false
}
#TOP_LEVEL_FLOW & {
    target: #PAIR_FLOW_TMPL & {bbox: #BBOX_LIST[0]}
}
