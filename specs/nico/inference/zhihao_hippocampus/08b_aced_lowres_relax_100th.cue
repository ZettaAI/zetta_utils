// #IMG_PATH: "gs://zetta-research-nico/hippocampus/low_res_enc_c4"
#IMG_PATH: "gs://zetta-research-nico/hippocampus/rigid_w_scale/low_res_enc_c4_rigid_100th"
#IMG_RES: [6144, 6144, 45]
#IMG_SIZE: [256, 256, 37]

#Z_OFFSETS: _ | *[-1]

// FOLDERS
#FOLDER:        "gs://zetta-research-nico/hippocampus"
#PROJECT:       "test-240307"
#BASE_FOLDER:   "gs://zetta-research-nico/hippocampus/aced_coarse_100th_w_scale/\(#PROJECT)"
#TMP_FOLDER:    "gs://tmp_2w/nico/aced_coarse_100th_w_scale"

#TEST_LOCAL: true

#CLUSTER_NUM_WORKERS: 16

#RUN_RELAX_FLOW:            false
//#RUN_POST_ALIGN_FLOW:       false
//#RUN_POST_ALIGN_WARP:       false


#IMG_WARP_OUTPUT_RES: [6144, 6144, #IMG_RES[2]]     // fast test output
#RELAXATION_RESOLUTION: [12288, 12288, #IMG_RES[2]]
#BLOCKS: [
    {_z_start: 0, _z_end: 23, _fix: "last"},
    {_z_start: 22, _z_end: #IMG_SIZE[2], _fix: "first"}
]
#RELAXATION_ITER: 20000
#RELAXATION_RIG:  1000
#RELAXATION_LR:   1e-4

// #MATCH_OFFSETS_FLOW: op_kwargs: tissue_mask: data_resolution: [6144, 6144, #IMG_RES[2]]
// #MATCH_OFFSETS_FLOW: op_kwargs: pairwise_fields: "-1": data_resolution: [6144, 6144, #IMG_RES[2]]
// #MATCH_OFFSETS_FLOW: op_kwargs: pairwise_fields_inv: "-1": data_resolution: [6144, 6144, #IMG_RES[2]]

// fields
#RELAX_FLOW: op_kwargs: pfields: "-1": data_resolution: [6144, 6144, #IMG_RES[2]]
#RELAX_FLOW: op_kwargs: rigidity_masks: data_resolution: [6144, 6144, #IMG_RES[2]]

// #RIGID_AFIELD_PATH: "gs://zetta-research-nico/hippocampus/rigid/field"
// #RELAX_FLOW: op_kwargs: first_section_fix_field: data_resolution: [1536, 1536, #IMG_RES[2]]
// #RELAX_FLOW: op_kwargs: last_section_fix_field: data_resolution: [1536, 1536, #IMG_RES[2]]


// Be more conservative with the misalignment outputs, set lower to be more conservative
#RELAX_OUTCOME_CHUNK_SIZE: [128, 128, 1]

// OTHER VARIABLES
#DEBUG_SUFFIX:      _ | *"_final"
#FOLDER:            _ | *"\(#BASE_FOLDER)/pair"
#TMP_PATH:          "\(#TMP_FOLDER)/\(#PROJECT)/\(#RELAXATION_SUFFIX)"
#FIELDS_PATH:       "\(#FOLDER)/pairwise/coarse_100th_w_scale_wo_rigid_inv/field_condensed"

#AFIELD_PATH:       "\(#FOLDER)/aced_coarse_100th_w_scale/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH:  _ | *"\(#FOLDER)/aced_coarse_100th_w_scale/img_aligned\(#RELAXATION_SUFFIX)"
#MATCH_OFFSET_BASE: "\(#FOLDER)/aced_coarse_100th_w_scale/match_offsets_\(#RELAXATION_RESOLUTION[0])nm"
#RELAXATION_SUFFIX: "_try_\(#RELAXATION_RESOLUTION[0])nm" +
                    "_iter\(#RELAXATION_ITER)" +
                    "_rig\(#RELAXATION_RIG)" +
                    "_lr\(#RELAXATION_LR)" +
                    "\(#DEBUG_SUFFIX)"

// #RIGID_AFIELD_PATH: _

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


// #MATCH_OFFSETS_FLOW: {
//     "@type": "build_subchunkable_apply_flow"
//     op: {
//         "@type": "AcedMatchOffsetOp"
//     }
//     bbox: _
//     processing_chunk_sizes: [[128, 128, bbox._z_end - bbox._z_start]]
//     processing_crop_pads: [[0, 0, 0]]
//     dst_resolution: #RELAXATION_RESOLUTION
//     // level_intermediaries_dirs: ["~/.zutils/tmp"]
//     skip_intermediaries: true
//     op_kwargs: {
//         max_dist: 1
//         tissue_mask: {
//             "@type": "build_cv_layer"
//             path:    #IMG_PATH
//             data_resolution?: _
//             interpolation_mode: "img"
// 			read_procs: [
//                 {"@type": "lambda", "lambda_str": "lambda x: x[0:1,:,:,:]"},
//                 {
//                     "@type": "compare"
//                     "@mode": "partial"
//                     mode:    "!="
//                     value:   0
//                 },
//                 {
//                     "@type": "filter_cc"
//                     "@mode": "partial"
//                     thr:     10
//                     mode:    "keep_large"
//                 },
//                 {
//                     // grow tissue mask by 10 px
//                     "@type": "kornia_dilation"
//                     "@mode": "partial"
//                     width:   21
//                 },
//                 {"@type": "to_uint8", "@mode": "partial"},
// 			]
//         }
//         misalignment_masks: {
//             for offset in #Z_OFFSETS {
//                 "\(offset)": {
//                     "@type": "build_constant_volumetric_layer"
//                     value: 0
//                 }
//             }
//         }
//         pairwise_fields: {
//             for offset in #Z_OFFSETS {
//                 "\(offset)": {
//                     "@type": "build_cv_layer"
//                     path:    "\(#FIELDS_PATH)"
//                     data_resolution?: _
//                     interpolation_mode: "field"
//                 }
//             }
//         }
//         pairwise_fields_inv: {
//             for offset in #Z_OFFSETS {
//                 "\(offset)": {
//                     "@type": "build_cv_layer"
//                     path:    "\(#FIELDS_PATH)"
//                     data_resolution?: _
//                     interpolation_mode: "field"
//                     read_procs: [
//                         { "@type": "invert_field", "@mode": "partial" }
//                     ]
//                 }
//             }
//         }
//     }
//     let match_offsets_path = "\(#MATCH_OFFSET_BASE)_\(bbox._z_start)_\(bbox._z_end)"
//     dst: {
//         "@type": "build_volumetric_layer_set"
//         layers: {
//             match_offsets: {
//                 "@type":             "build_cv_layer"
//                 path:                match_offsets_path
//                 info_reference_path: #IMG_PATH
//                 info_chunk_size:     #RELAX_OUTCOME_CHUNK_SIZE
//                 info_add_scales:     [dst_resolution]
//                 info_add_scales_mode: "replace"
//                 info_field_overrides: {
//                     num_channels: 1
//                     data_type:    "uint8"
//                 }
//                 on_info_exists:      "overwrite"
//                 write_procs: [
//                     {"@type": "to_uint8", "@mode": "partial"},
//                 ]
//             }
//             img_mask: {
//                 "@type":             "build_cv_layer"
//                 path:                "\(match_offsets_path)/img_mask"
//                 info_reference_path: #IMG_PATH
//                 info_chunk_size:     #RELAX_OUTCOME_CHUNK_SIZE
//                 info_add_scales:     [dst_resolution]
//                 info_add_scales_mode: "replace"
//                 info_field_overrides: {
//                     num_channels: 1
//                     data_type:    "uint8"
//                 }
//                 on_info_exists:      "overwrite"
//                 write_procs: [
//                     {"@type": "filter_cc", "@mode": "partial", thr: 3, mode: "keep_large"},
//                     {"@type": "to_uint8", "@mode":  "partial"},
//                 ]
//             }
//             aff_mask: {
//                 "@type":             "build_cv_layer"
//                 path:                "\(match_offsets_path)/aff_mask"
//                 info_reference_path: #IMG_PATH
//                 info_chunk_size:     #RELAX_OUTCOME_CHUNK_SIZE
//                 info_add_scales:     [dst_resolution]
//                 info_add_scales_mode: "replace"
//                 info_field_overrides: {
//                     num_channels: 1
//                     data_type:    "uint8"
//                 }
//                 on_info_exists:      "overwrite"
//                 write_procs: [
//                     {"@type": "filter_cc", "@mode": "partial", thr: 3, mode: "keep_large"},
//                     {"@type": "to_uint8", "@mode":  "partial"},
//                 ]
//             }
//             sector_length_before: {
//                 "@type":             "build_cv_layer"
//                 path:                "\(match_offsets_path)/sl_before"
//                 info_reference_path: #IMG_PATH
//                 info_chunk_size:     #RELAX_OUTCOME_CHUNK_SIZE
//                 info_add_scales:     [dst_resolution]
//                 info_add_scales_mode: "replace"
//                 info_field_overrides: {
//                     num_channels: 1
//                     data_type:    "uint8"
//                 }
//                 on_info_exists:      "overwrite"
//                 write_procs: [
//                     {"@type": "to_uint8", "@mode": "partial"},
//                 ]
//             }
//             sector_length_after: {
//                 "@type":             "build_cv_layer"
//                 path:                "\(match_offsets_path)/sl_after"
//                 info_reference_path: #IMG_PATH
//                 info_chunk_size:     #RELAX_OUTCOME_CHUNK_SIZE
//                 info_add_scales:     [dst_resolution]
//                 info_add_scales_mode: "replace"
//                 info_field_overrides: {
//                     num_channels: 1
//                     data_type:    "uint8"
//                 }
//                 on_info_exists:      "overwrite"
//                 write_procs: [
//                     {"@type": "to_uint8", "@mode": "partial"},
//                 ]
//             }
//         }
//     }
// }

#RELAX_FLOW: {
    "@type": "build_subchunkable_apply_flow"
    op: {
        "@type": "AcedRelaxationOp"
    }
    expand_bbox_processing:    true
    dst_resolution: #RELAXATION_RESOLUTION
    bbox:           _
    processing_chunk_sizes: [[128, 128, bbox._z_end - bbox._z_start]]
    max_reduction_chunk_sizes: [128, 128, bbox._z_end - bbox._z_start]
    processing_crop_pads: [[0, 0, 0]]
    processing_blend_pads: [[0, 0, 0]]
    // level_intermediaries_dirs: [#TMP_PATH]
    skip_intermediaries: true
    op_kwargs: {
        max_dist: 1
        fix:                     _
        num_iter:                #RELAXATION_ITER
        lr:                      #RELAXATION_LR
        rigidity_weight:         #RELAXATION_RIG
        rigidity_masks: {
            "@type": "build_cv_layer"
            path:    #IMG_PATH
            data_resolution?: _
            interpolation_mode: "img"
			read_procs: [
                {"@type": "lambda", "lambda_str": "lambda x: x[0:1,:,:,:]"},
                {
                    "@type": "compare"
                    "@mode": "partial"
                    mode:    "!="
                    value:   0
                },
                {
                    "@type": "filter_cc"
                    "@mode": "partial"
                    thr:     10
                    mode:    "keep_large"
                },
                {
                    // grow tissue mask by 10 px
                    "@type": "kornia_dilation"
                    "@mode": "partial"
                    width:   21
                },
                {"@type": "to_uint8", "@mode": "partial"},
			]
        }

        match_offsets: {
            "@type": "build_constant_volumetric_layer"
            value: 1
        }
        // let match_offsets_path = "\(#MATCH_OFFSET_BASE)_\(bbox._z_start)_\(bbox._z_end)"
        // match_offsets: {
        //     "@type": "build_cv_layer"
        //     path:    match_offsets_path
        //     //info_reference_path: #IMG_PATH
        //     // on_info_exists: "overwrite"
        // }
        pfields: {
            for offset in #Z_OFFSETS {
                "\(offset)": {
                    "@type": "build_cv_layer"
                    path:    "\(#FIELDS_PATH)"
                    data_resolution?: _
                    interpolation_mode: "field"
                }
            }
        }
        // first_section_fix_field: {
        //     "@type": "build_cv_layer"
        //     path:    #RIGID_AFIELD_PATH
        //     data_resolution?: _
        //     interpolation_mode: "field"
        // }
        // last_section_fix_field: {
        //     "@type": "build_cv_layer"
        //     path:    #RIGID_AFIELD_PATH
        //     data_resolution?: _
        //     interpolation_mode: "field"
        // }
    }
    dst: {
        "@type":             "build_cv_layer"
        path:                #AFIELD_PATH
        info_reference_path: #IMG_PATH
        info_field_overrides: {
            num_channels: 2
            data_type:    "float32"
        }
        info_add_scales: [#RELAXATION_RESOLUTION]
        info_add_scales_mode: "replace"
        info_chunk_size: #RELAX_OUTCOME_CHUNK_SIZE
        on_info_exists:  "overwrite"
    }
}


#IMG_WARP_DOWNSAMPLING: _ | *[1, 1, 1]
#RUN_POST_ALIGN_WARP:     _ | *true

#POST_ALIGN_FLOW: {
    _bbox:   _
    "@type": "mazepa.concurrent_flow"
    stages: [
        if #RUN_POST_ALIGN_WARP {
            #WARP_FLOW_TMPL & {
                bbox: _bbox
                op: mode: "img"
                op: downsampling_factor: #IMG_WARP_DOWNSAMPLING
                op_kwargs: src: path:              #IMG_PATH
                op_kwargs: field: path:            #AFIELD_PATH
                op_kwargs: field: data_resolution: #RELAXATION_RESOLUTION
                op_kwargs: field: interpolation_mode: "field"
                dst: path: #IMG_ALIGNED_PATH
                dst: info_add_scales: [#IMG_WARP_OUTPUT_RES]
                dst: info_add_scales_mode: "replace"
                dst_resolution: #IMG_WARP_OUTPUT_RES
            }
        }
    ]
}

#WARP_FLOW_TMPL: {
    "@type": "build_subchunkable_apply_flow"
    op: {
        "@type": "WarpOperation"
        mode:    _
        downsampling_factor?: _
    }
    processing_chunk_sizes: [[256,256,1]]
    processing_crop_pads: [[0, 0, 0]]
    skip_intermediaries: true
    // level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
    //chunk_size: [512, 512, 1]
    bbox:           _
    dst_resolution: _ | *[6144, 6144, #IMG_RES[2]]
    op_kwargs: {
        src: {
            "@type":      "build_cv_layer"
            path:         _
            read_procs?:  _
            index_procs?: _ | *[]
        }
        field: {
            "@type":            "build_cv_layer"
            path:               _
            data_resolution:    _ | *null
            interpolation_mode: "field"
        }
    }
    dst: {
        "@type":             "build_cv_layer"
        path:                _
        info_reference_path: #IMG_PATH
        info_add_scales?: _
        info_add_scales_mode?: _
        on_info_exists:      "overwrite"
        write_procs?:        _
        index_procs?:        _ | *[]
    }
}

#TEST_LOCAL: _ | *false
#RUN_OFFSET:             _ | *true
#RUN_RELAX_FLOW:         _ | *true
#RUN_POST_ALIGN_FLOW:    _ | *true
#RUN_INFERENCE: {
    "@type":      "mazepa.execute_on_gcp_with_sqs"
    worker_cluster_region:  "us-east1"
    worker_cluster_project: "zetta-research"
    worker_cluster_name:    "zutils-x3"
    worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:nico_py3.9_20240306"
    worker_resources: {
        "nvidia.com/gpu"?: _
    }
    worker_resource_requests: {
        memory: "13000Mi"       // sized for n1-highmem-4
    }
    worker_replicas:     #CLUSTER_NUM_WORKERS
    local_test:          #TEST_LOCAL
    if #TEST_LOCAL == true {
        debug: true
    }

    target: {
        "@type": "mazepa.concurrent_flow"
        stages: [
            for block in #BLOCKS {
                let bbox = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
                "@type": "mazepa.sequential_flow"
                stages: [
                    // if #RUN_OFFSET {
                    //     #MATCH_OFFSETS_FLOW & {'bbox': bbox},
                    // }
                    if #RUN_RELAX_FLOW {
                        #RELAX_FLOW & {'bbox': bbox, op_kwargs: fix: block._fix},
                    }
                    if #RUN_POST_ALIGN_FLOW {
                        #POST_ALIGN_FLOW & {_bbox: bbox},
                    }
                ]
            },
        ]
    }
}
#RUN_INFERENCE
if #RUN_RELAX_FLOW | #RUN_POST_ALIGN_FLOW {
    #RUN_INFERENCE: worker_resources: "nvidia.com/gpu": "1"
}