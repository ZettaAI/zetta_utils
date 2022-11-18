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
	"@type":      "chunked_interpolate_xy"
	scale_factor: 0.5
	chunker: {
		"@type": "VolumetricIndexChunker"
		"chunk_size": [4 * 1024, 4 * 1024, 1]
		"step_size": [4 * 1024, 4 * 1024, 1]
	}
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mode: "img"
	idx:  #IDX
}
