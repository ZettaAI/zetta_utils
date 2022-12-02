#VERSION: "x1_masked"

#V:            "4_both"
#STAGE_PREFIX: "256_128_64_32nm"
#SRC_PATH:     "gs://zfish_unaligned/coarse_x0/raw_img"
#AFIELD_PATH:  "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/relaxed/afield_\(#VERSION)_v\(#V)"
#DST_PATH:     "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/relaxed/raw_img_warped_\(#VERSION)_v\(#V)"
#DST_RESOLUTION: [32, 32, 30]

#XY_CROP:      64
#XY_OUT_CHUNK: 2048

"@type": "mazepa_execute"
target: {
	"@type":        "build_warp_flow"
	mode:           "img"
	dst_resolution: #DST_RESOLUTION
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 - 512, 1024 - 256, 3000]
		end_coord: [1024, 1024 + 256, 3020]
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
		path:    #AFIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "expect_same"
	}
}
