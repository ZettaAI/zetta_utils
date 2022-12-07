#SRC_PATH:   "gs://tmp_2w/inference_tests/raw_img_x20"
#FIELD_PATH: "gs://tmp_2w/inference_tests/field_x20"
#DST_PATH:   "gs://tmp_2w/inference_tests/warped_img_x20"

#XY_CROP:      512
#XY_OUT_CHUNK: 1024

"@type": "mazepa.execute"
target: {
	"@type": "build_warp_flow"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 7, 1024 * 3, 2001]
		end_coord: [1024 * 9, 1024 * 5, 2003]
		resolution: [64, 64, 40]
	}
	dst_resolution: [64, 64, 40]
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
