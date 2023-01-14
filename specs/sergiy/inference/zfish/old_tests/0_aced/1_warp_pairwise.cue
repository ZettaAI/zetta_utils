#VERSION:  "x2_masked"
#RIGIDITY: 50
#NUM_ITER: 100

#STAGE_PREFIX: "256_128_64_32nm"
#SRC_PATH:     "gs://zfish_unaligned/coarse_x0/encodings_masked"
#FIELD_PATH:   "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
#DST_PATH:     "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/encodings_warped_to_z-1_\(#VERSION)"
#DST_RESOLUTION: [32, 32, 30]

#XY_CROP:      512
#XY_OUT_CHUNK: 2048

"@type": "mazepa.execute"
target: {
	"@type":        "build_warp_flow"
	mode:           "img"
	dst_resolution: #DST_RESOLUTION
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3001]
		end_coord: [2048, 2048, 3002]
		resolution: [256, 256, 30]
	}
	crop: [#XY_CROP, #XY_CROP, 0]
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "expect_same"
	}
}
