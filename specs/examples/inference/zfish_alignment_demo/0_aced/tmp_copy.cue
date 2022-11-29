#STAGE_PREFIX: "256_128_64_32nm"

#SRC_PATH: "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_x2_masked"
#DST_PATH: "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_x1_masked"

#DST_RESOLUTION: [32, 32, 30]

#XY_OUT_CHUNK: 2048

"@type": "mazepa_execute"
target: {
	"@type":        "build_write_flow"
	dst_resolution: #DST_RESOLUTION
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3010]
		end_coord: [2048, 2048, 3020]
		resolution: [256, 256, 30]
	}
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type": "build_cv_layer"
		path:    #DST_PATH
	}
}
