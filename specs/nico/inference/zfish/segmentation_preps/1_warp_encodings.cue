#TMP_PATH: "gs://tmp_2w/temporary_layers"

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "mask"
	}
	expand_bbox: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	max_reduction_chunk_sizes: [1024 * 8, 1024 * 8, 1]
	processing_crop_pads: [[0, 0, 0], [256, 256, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	bbox:           _
	dst_resolution: [32, 32, 30]
    op_kwargs: {
        src: {
            "@type":      "build_ts_layer"
            path:         "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip3_aced_encodings"
            read_procs: [
                {
                    "@type": "compare"
                    "@mode": "partial"
                    mode:    "!="
                    value:   0
                }
            ],
            index_procs: [
                {
                    "@type": "VolumetricIndexTranslator"
                    offset: _
                    resolution: [32, 32, 30]
                }
            ]
        }
        field: {
            "@type":            "build_cv_layer"
            path:               "gs://zetta_jlichtman_zebrafish_001_alignment_temp/aced/med_x1/final_x0/afieldtry_x8_512nm_iter8000_rig0.5_lr0.001_clip0.01"
            data_resolution:    [512,512,30]
            interpolation_mode: "field"
            index_procs: op_kwargs.src.index_procs
        }
    }
	dst: {
		"@type":             "build_cv_layer"
		path:                "gs://zetta_jlichtman_zebrafish_001_alignment_temp/aced/med_x1/final_x0/warped_enc"
		info_reference_path: "gs://zetta_jlichtman_zebrafish_001_alignment/fine_full_v2/img"
		on_info_exists:      "overwrite"
		write_procs:        [
            {
                "@type": "to_uint8"
                "@mode": "partial"
            }
        ]
	}
}


"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-jlichtman-zebrafish-001/zetta_utils:nico_py3.9_20230517"
worker_resources: {
	memory:           "27560Mi"
	// "nvidia.com/gpu": "1"
}
worker_replicas:      100
batch_gap_sleep_sec:  1
do_dryrun_estimation: true
local_test:           false
worker_cluster_name:    "zutils-zfish"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-jlichtman-zebrafish-001"
target: {
    "@type": "mazepa.concurrent_flow"
    stages: [
        #WARP_FLOW_TMPL & {
            op_kwargs: src: index_procs: [
                {
				    "@type": "VolumetricIndexTranslator"
					offset: [0, 0, 0]
					resolution: [32, 32, 30]
				}
            ],
            bbox: {
                "@type": "BBox3D.from_coords"
                start_coord: [0, 0, 0]
                end_coord: [12288, 16384, 158]
                resolution: [32, 32, 30]
            }
        },
        #WARP_FLOW_TMPL & {
            op_kwargs: src: index_procs: [
                {
				    "@type": "VolumetricIndexTranslator"
					offset: [0, 0, 1]
					resolution: [32, 32, 30]
				}
            ],
            bbox: {
                "@type": "BBox3D.from_coords"
                start_coord: [0, 0, 158]
                end_coord: [12288, 16384, 278]
                resolution: [32, 32, 30]
            }
        },
        #WARP_FLOW_TMPL & {
            op_kwargs: src: index_procs: [
                {
				    "@type": "VolumetricIndexTranslator"
					offset: [0, 0, 3]
					resolution: [32, 32, 30]
				}
            ],
            bbox: {
                "@type": "BBox3D.from_coords"
                start_coord: [0, 0, 278]
                end_coord: [12288, 16384, 2228]
                resolution: [32, 32, 30]
            }
        },
        #WARP_FLOW_TMPL & {
            op_kwargs: src: index_procs: [
                {
				    "@type": "VolumetricIndexTranslator"
					offset: [0, 0, 4]
					resolution: [32, 32, 30]
				}
            ],
            bbox: {
                "@type": "BBox3D.from_coords"
                start_coord: [0, 0, 2228]
                end_coord: [12288, 16384, 4010]
                resolution: [32, 32, 30]
            }
        }
    ]
}