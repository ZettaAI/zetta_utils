#SRC_PATH:  "gs://zfish_unaligned/coarse_x0/test_x0/encodings_x1"
#MASK_PATH: "gs://zfish_unaligned/coarse_x0/defect_mask"
#DST_PATH:  "gs://zfish_unaligned/coarse_x0/test_x0/encodings_x1_masked"

#XY_CROP:      256
#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_apply_mask_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: _
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mask: {
		"@type": "build_cv_layer"
		path:    #MASK_PATH
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
		start_coord: [0, 0, 0]
		end_coord: [2048, 2048, 25]
		resolution: [512, 512, 30]
	}
}

#RESOLUTIONS: [
	[1024, 1024, 30],
	[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	//[64, 64, 30],
	//[32, 32, 30],
]

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x26"
worker_replicas: 10
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in #RESOLUTIONS {
			#FLOW_TMPL & {dst_resolution: res}
		},
	]
}
