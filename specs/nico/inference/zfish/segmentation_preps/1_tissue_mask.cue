#IMG_PATH:           "gs://zetta_jlichtman_zebrafish_001_alignment_temp/fine_full_v2/masks/img_close9"
#IMG_PATH_3X:        "gs://zetta_jlichtman_zebrafish_001_alignment_temp/fine_full_v2/masks/img_close9_open5"
#TMP_PATH:           "gs://tmp_2w/temporary_layers"

#DATASET_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [10240, 16384, 4010]
	resolution: [32, 32, 30]
}

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [int, int, int] | *[10240, 16384, 4010]
	resolution: [32, 32, 30]
}

#CLOSING_TEMPLATE: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "kornia_opening"
			"@mode": "partial"
			kernel: {
				"@type": "torch.ones"
				// "@mode":    "partial"
				size: [1, 5]
			}
		}
	}
	expand_bbox: true
	processing_chunk_sizes: [[1280 * 2, 1024 * 2, 20]]
	processing_crop_pads: [[0, 0, 3]]
	level_intermediaries_dirs: ["~/.zutils/tmp"]
	dst_resolution: [128, 128, 30]
	bbox: #ROI_BOUNDS
	op_kwargs: {
		data: {
			"@type": "build_ts_layer"
			path:    #IMG_PATH
			read_procs: [
				{
					"@mode":   "partial"
					"@type":   "rearrange"
					"pattern": "c x y z -> x y z c"  // Want to operate on yz axis (xz would also work)
				},
				{
					"@type": "compare"
					"@mode": "partial"
					mode:    "!="
					value:   0
				},
				{
					"@type": "to_uint8"
					"@mode": "partial"
				}
			]
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #IMG_PATH_3X
		info_reference_path: #IMG_PATH
		on_info_exists:      "overwrite"
		info_field_overrides: {
			data_type: "uint8"
		}
		write_procs: [
			{
				"@mode":   "partial"
				"@type":   "rearrange"
				"pattern": "x y z c -> c x y z"
			}
		]
	}
}


"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-jlichtman-zebrafish-001/zetta_utils:nico_py3.9_20230630"
worker_resources: {
	memory: "27560MiB"
}
worker_replicas:        50
batch_gap_sleep_sec:    1
do_dryrun_estimation:   true
local_test:             false
worker_cluster_name:    "zutils-zfish"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-jlichtman-zebrafish-001"

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#CLOSING_TEMPLATE,
	]
}
