#SRC_PATH:  "gs://zfish_unaligned/coarse_x0/encodings"
#MASK_PATH: "gs://zfish_unaligned/coarse_x0/defect_mask"
#DST_PATH:  "gs://zfish_unaligned/coarse_x0/encodings_masked"

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
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "expect_same"
	}
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3010]
		end_coord: [2048, 2048, 3020]
		resolution: [512, 512, 30]
	}
}

#RESOLUTIONS: [
	[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

"@type": "mazepa_execute"
target: [
	for res in #RESOLUTIONS {
		#FLOW_TMPL & {dst_resolution: res}
	},
]
