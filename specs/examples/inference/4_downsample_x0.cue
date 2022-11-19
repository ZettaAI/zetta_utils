#SRC_PATH: "gs://tmp_2w/inference_tests/raw_img_x0"
#IDX: {
	"@type": "VolumetricIndex"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 2, 1024 * 0, 2000]
		end_coord: [1024 * 10, 1024 * 8, 2005]
		resolution: [64, 64, 40]
	}
	resolution: [64, 64, 40]
}
"@type": "mazepa_execute"
target: {
	"@type": "chunked_interpolate"
	dst_res: [128, 128, 40]
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [2 * 1024, 2 * 1024, 1]
		step_size: [2 * 1024, 2 * 1024, 1]
		resolution: [128, 128, 40]
	}
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mode: "img"
	idx:  #IDX
}
