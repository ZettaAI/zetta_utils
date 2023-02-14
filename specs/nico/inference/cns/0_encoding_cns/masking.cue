#DEFECTS_PATH: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
#RESIN_PATH:   "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/resin_mask"

#IMG_PATH:        "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/elastic_m3_m9_v1"
#IMG_MASKED_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/elastic_m3_m9_v1_masked"

#BBOX: {
    "@type": "BBox3D.from_coords"
    start_coord: [0, 0, 3300]
    end_coord: [2048, 2048, 3501]
    resolution: [512, 512, 45]
}
#BIG_BBOX: {
    "@type": "BBox3D.from_coords"
    start_coord: [0, 0, 3300]
    end_coord: [1024 * 8, 1024 * 8, 3501]
    resolution: [512, 512, 45]
}

#FLOW_TMPL: {
    "@type":        "build_apply_mask_flow"
    chunk_size:     _
    dst_resolution: _
    src: {
        "@type": "build_cv_layer"
        path:    _
    }
    masks: [
        {
            "@type": "build_cv_layer"
            path:    #DEFECTS_PATH
            read_procs: [
                {
                    "@type": "coarsen_mask"
                    "@mode": "partial"
                    width:   1
                },

            ]
        },
        {
            "@type": "build_cv_layer"
            path:    #RESIN_PATH
            data_resolution: [256, 256, 45]
            interpolation_mode: "mask"
        },
    ]

    dst: {
        "@type":             "build_cv_layer"
        path:                _
        info_reference_path: src.path
        on_info_exists:      "expect_same"
    }
    bbox: _
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_masks"
worker_resources: {
    memory: "18560Mi"
}
worker_replicas:     100
batch_gap_sleep_sec: 0.05

local_test: false

target: {
    "@type": "mazepa.concurrent_flow"
    stages: [
        for res in [32, 64, 128] {
            #FLOW_TMPL & {
                chunk_size: [1024 * 4, 1024 * 4, 1]
                bbox: #BBOX
                dst_resolution: [res, res, 45]
                src: path: #IMG_PATH
                dst: path: #IMG_MASKED_PATH
            }
        },
        for res in [256, 512] {
            #FLOW_TMPL & {
                chunk_size: [1024 * 2, 1024 * 2, 1]
                bbox: #BBOX
                dst_resolution: [res, res, 45]
                src: path: #IMG_PATH
                dst: path: #IMG_MASKED_PATH
            }
        },
        for res in [1024, 2048, 4096] {
            #FLOW_TMPL & {
                chunk_size: [1024, 1024, 1]
                bbox: #BIG_BBOX
                dst_resolution: [res, res, 45]
                src: path: #IMG_PATH
                dst: path: #IMG_MASKED_PATH
            }

        },

    ]
}