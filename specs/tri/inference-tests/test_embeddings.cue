#IMG_PATH: "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg"
#IMG_RES: [8, 8, 8]
#IMG_SIZE: [34428, 39726, 41394]

#DST_PATH: "gs://tmp_2w/tri/test_embeddings-8nm"

#MODEL_RES_XY: 8
// #MODEL_RES_XY: 16
#USE_BACKGROUND_MASK: false

if #MODEL_RES_XY == 8 {
    #MODEL_PATH: "gs://zetta-research-kisuk/isometric/DeepEM/experiments/230920-metric-learning-aniso-x0/models/model280000.onnx"
    #MODEL_INPUT_SIZE: [160, 160, 160]
    #MODEL_OUTPUT_SIZE: [128, 128, 128]
    #MODEL_CROP_PADS: [16, 16, 16]

    #PROCESS_BLEND_PADS: [0, 0, 0]
    #PROCESS_CROP_PADS: [6, 6, 6]
    #PROCESS_CHUNK_SIZE: [116, 116, 116]

    // #INFERENCE_TMPL: op: fn: bg_mask_channel: 0
    // #INFERENCE_TMPL: op: fn: bg_mask_threshold: 0.2
    #MODEL_NUM_CHANNELS: 4  // get only the first 4 channels
    #INFERENCE_TMPL: op: fn: output_channels: [1, 2, 3, 4]
}
if #MODEL_RES_XY == 16 {
    #MODEL_PATH: "gs://zetta-research-kisuk/isometric/DeepEM/experiments/230920-metric-learning-aniso-x0/models/16nm/model280000.onnx"
    #MODEL_INPUT_SIZE: [80, 80, 80]
    #MODEL_OUTPUT_SIZE: [64, 64, 64]
    #MODEL_CROP_PADS: [8, 8, 8]

    #PROCESS_BLEND_PADS: [0, 0, 0]
    #PROCESS_CROP_PADS: [3, 3, 3]

    // #INFERENCE_TMPL: op: fn: bg_mask_channel: 24
    // #INFERENCE_TMPL: op: fn: bg_mask_threshold: 0.2
    #MODEL_NUM_CHANNELS: 4  // get only the first 4 channels
    #INFERENCE_TMPL: op: fn: output_channels: [0, 1, 2, 3]
}

// 16384, 20480, 18944
#TEST_SMALL: true
#TEST_LOCAL: true
if #TEST_SMALL {
    // ng coord: 16384, 20480, 18944 in 8nm
    let test_grid_size = [3, 3, 3]
    let cs = #PROCESS_CHUNK_SIZE
    #BLOCKS: [
        {_z_start: 18944,   _z_end: _z_start+cs[2]*1*test_grid_size[2]},
    ]
    #START_COORD_XY:    [16384, 20480]
    #END_COORD_XY:      [#START_COORD_XY[0]+cs[0]*1*test_grid_size[0],
                         #START_COORD_XY[1]+cs[1]*1*test_grid_size[1]]
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
                    dst_resolution: [#MODEL_RES_XY, #MODEL_RES_XY, #MODEL_RES_XY]
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
        info_field_overrides: num_channels: #MODEL_NUM_CHANNELS
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
    info_add_scales: [[#MODEL_RES_XY, #MODEL_RES_XY, #MODEL_RES_XY]]
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
