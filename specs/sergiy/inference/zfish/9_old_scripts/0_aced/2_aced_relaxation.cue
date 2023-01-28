#VERSION: "x1_masked"

#V:            "4_both"
#STAGE_PREFIX: "256_128_64_32nm"
#SRC_PATH:     "gs://zfish_unaligned/coarse_x0/encodings_masked"
#PFIELD_PATH:  "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
#DST_PATH:     "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/relaxed/afield_\(#VERSION)_v\(#V)"

#DST_RESOLUTION: [32, 32, 30]

#XY_CROP:      128
#XY_OUT_CHUNK: 2048

"@type": "mazepa.execute"
target: {
	"@type":         "build_aced_relaxation_flow"
	dst_resolution:  #DST_RESOLUTION
	rigidity_weight: 3
	num_iter:        150
	fix:             "both"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [1024 - 512, 1024 - 256, 3010]
		end_coord: [1024, 1024 + 256, 3020]
		resolution: [256, 256, 30]
	}
	crop: [#XY_CROP, #XY_CROP, 0]
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 10]
	field: {
		"@type": "build_cv_layer"
		path:    #PFIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #PFIELD_PATH
		on_info_exists:      "expect_same"
	}
}
