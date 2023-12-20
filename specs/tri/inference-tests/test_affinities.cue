#IMG_PATH: "precomputed://gs://lee-mouse-spinal-cord-001-raw/raw"
#IMG_RES: [4, 4, 45]
#IMG_SIZE: [417792, 200704, 4216]

#DST_PATH: "gs://tmp_2w/tri/test_affinities-bg"

#MODEL_PATH: "gs://lee-mouse-spinal-cord-001-binaries/affinity_pretrained_minnie_mye.onnx"
#MODEL_RES_XY: 8
#MODEL_INPUT_SIZE: [256, 256, 20]
#MODEL_OUTPUT_SIZE: [192, 192, 16]
#MODEL_CROP_PADS: [32, 32, 2]

#INFERENCE_TMPL: op: fn: output_channels: [0, 1, 2]
#APPLY_BACKGROUND_MASK: true
// #APPLY_BACKGROUND_MASK: false
if #APPLY_BACKGROUND_MASK {
    #INFERENCE_TMPL: op: fn: bg_mask_channel: 3
    #INFERENCE_TMPL: op: fn: bg_mask_threshold: 0.2
}

// Blend/crop parameters
#PROCESS_BLEND_PADS: [16, 16, 0]
#PROCESS_CROP_PADS: [0, 0, 0]
#PROCESS_CHUNK_SIZE: [160, 160, 16]

#TEST_SMALL: true
#TEST_LOCAL: true
if #TEST_SMALL {
    // ng coord: 208896, 37888, 2944 in 4nm
    let test_grid_size = [3, 3, 3]
    let cs = #PROCESS_CHUNK_SIZE
    #BLOCKS: [
        {_z_start: 2944,   _z_end: _z_start+cs[2]*test_grid_size[2]},
    ]
    #START_COORD_XY:    [208896, 37888]
    #END_COORD_XY:      [#START_COORD_XY[0]+cs[0]*2*test_grid_size[0],
                         #START_COORD_XY[1]+cs[1]*2*test_grid_size[1]]
}

#BBOX_TMPL: {
    "@type":  "BBox3D.from_coords"
    _z_start: int
    _z_end:   int
    start_coord: [#START_COORD_XY[0], #START_COORD_XY[1], _z_start]
    end_coord: [#END_COORD_XY[0], #END_COORD_XY[1], _z_end]
    resolution: #IMG_RES
}
#START_COORD_XY: _ | *[0, 0]
#END_COORD_XY: _ | *[#IMG_SIZE[0], #IMG_SIZE[1]]
#BLOCKS: _ | *[
    {_z_start: 0,   _z_end: #IMG_SIZE[2]},
]

#SRC: {
    "@type":  "build_cv_layer"
    path:      #IMG_PATH
}

#TEST_LOCAL: _ | false
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
        cuda: 3
        cpu: num_procs
    }
    target: _
}

#GCP_FLOW: {
    "@type":      "mazepa.execute_on_gcp_with_sqs"
    worker_cluster_region:  "us-east1"
    // worker_image:           "us.gcr.io/zetta-research/zetta_utils:tri-test-230829"
    // worker_cluster_project: "zetta-research"
    // worker_cluster_name:    "zutils-x3"
    worker_image:           "us.gcr.io/zetta-jkim-001/zetta_utils:tri-230830"
    worker_cluster_project: "zetta-jkim-001"
    worker_cluster_name:    "zutils"
    worker_resources: {
        memory: "21000Mi"       // sized for n1-highmem-4
        "nvidia.com/gpu": "1"
    }
    worker_replicas:     32
    local_test:          #TEST_LOCAL
    target: _
}

#PROCESS_CHUNK_SIZE: _
#PROCESS_BLEND_PADS: _
#PROCESS_CROP_PADS: _
#TOP_LEVEL_FLOW & {
    target: {
        "@type": "mazepa.concurrent_flow"
        stages: [
            for block in #BLOCKS {
                let bbox_ = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
                #INFERENCE_TMPL & {
                    bbox: bbox_
                    op: fn: model_path:     #MODEL_PATH
                    op: crop_pad:           #MODEL_CROP_PADS
                    processing_chunk_sizes: [#PROCESS_CHUNK_SIZE]
                    processing_blend_pads:  [#PROCESS_BLEND_PADS]
                    processing_crop_pads:   [#PROCESS_CROP_PADS]
                    dst_resolution: [#MODEL_RES_XY, #MODEL_RES_XY, #IMG_RES[2]]
                }
            }
        ]
    }
}

#MODEL_PATH: _
#MODEL_INPUT_SIZE: _
#MODEL_OUTPUT_SIZE: _
#MODEL_CROP_PADS: _
#MODEL_NUM_CHANNELS: _
#VEC2AFF_FUNCTION: _
#INFERENCE_TMPL: {
    "@type": "build_subchunkable_apply_flow"
    bbox: _
    // level_intermediaries_dirs: ["file://~/.zetta_utils/tmp/"]
    level_intermediaries_dirs: ["gs://tmp_2w/tri"]
    // skip_intermediaries: true
    op: {
        "@type": "VolumetricCallableOperation"
        fn: {
            "@type": "AffinitiesInferencer"
            model_path: _
            output_channels: _
            mask_channel?: _
            mask_threshold?: _
            mask_invert_threshold?: _
        }
        fn_semaphores: ["cuda"]
        crop_pad?: _
    }
    op_kwargs: {
        image: #SRC
        image_mask: _ | *#EMPTY_MASK_ONES
        output_mask: _ | *#EMPTY_MASK_ONES
    }
    dst_resolution: _
    processing_chunk_sizes: _
    processing_crop_pads?: _
    processing_blend_pads?: _
    // expand_bbox_resolution: true

    dst: #DATASET_TMPL & {
        path: #DST_PATH
        info_field_overrides: num_channels: 3
        write_procs: [
            // write as uint8
            {"@type": "multiply", "@mode": "partial", value: 255},
            {"@type": "to_uint8", "@mode": "partial"},
        ]
    }
}
#EMPTY_MASK_ONES: {
        "@type": "build_constant_volumetric_layer"
        value: 1
}

#DATASET_TMPL: {
    "@type": "build_cv_layer"
    path: _
    info_add_scales_ref: {
        resolution: #IMG_RES
        size: #IMG_SIZE
        chunk_sizes: [#PROCESS_CHUNK_SIZE]
        encoding: "raw"
        voxel_offset: [0, 0, 0]
    }
    info_add_scales: [[#MODEL_RES_XY, #MODEL_RES_XY, #IMG_RES[2]]]
    info_add_scales_mode: "replace"
    info_field_overrides: {
        type:         "image"
        num_channels: _
        data_type:    "uint8"
        type: "image"
    }
    on_info_exists:      "overwrite"
    write_procs?: _
}
