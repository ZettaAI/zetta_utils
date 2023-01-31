#SRC_PATH:  "gs://zfish_unaligned/precoarse_x0/test_x0/encodings_x1"
#MASK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"
#DST_PATH:  "gs://zfish_unaligned/precoarse_x0/test_x0/encodings_x1_masked_debug_x0"

#XY_CROP:      256
#XY_OUT_CHUNK: 1024

#FLOW: {
	"@type": "build_apply_mask_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: [512, 512, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mask: {
		"@type": "build_cv_layer"
		path:    #MASK_PATH
		data_resolution: [512, 512, 30]
		interpolation_mode: "mask"
		read_procs: [
			{
				"@type": "coarsen_mask"
				"@mode": "partial"
				width:   1
			},

		]

	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "expect_same"
	}
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [0, 0, 30]
		end_coord: [2048, 2048, 33]
		resolution: [512, 512, 30]
	}
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x26"
worker_replicas: 0
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: true

target: #FLOW
