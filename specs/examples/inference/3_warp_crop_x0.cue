#SRC_PATH:   "gs://tmp_2w/inference_tests/raw_img_x0"
#FIELD_PATH: "gs://tmp_2w/inference_tests/field_x0/neighbor_field_z_-1"
#DST_PATH:   "gs://tmp_2w/inference_tests/warped_img_x0"

#XY_OVERLAP:   512
#XY_OUT_CHUNK: 1024

#IDX: {
	"@type": "VolumetricIndex"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [-#XY_OVERLAP/2 + 1024*7, -#XY_OVERLAP/2 + 1024*3, 2001]
		end_coord: [#XY_OVERLAP + 1024*9, #XY_OVERLAP + 1024*5, 2003]
		resolution: [64, 64, 40]
	}
	resolution: [64, 64, 40]
}

"@type": "mazepa_execute"
target: {
	"@type": "build_chunked_apply_flow"
	task_factory: {
		"@type": "WarpTaskFactory"
		dst_data_crop: [#XY_OVERLAP / 2, #XY_OVERLAP / 2, 0]
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		"chunk_size": [#XY_OUT_CHUNK + #XY_OVERLAP, #XY_OUT_CHUNK + #XY_OVERLAP, 1]
		"step_size": [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	}
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_postprocs: [
			{
				"@type": "to_float32"
				"@mode": "partial"
			},
		]
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
		write_preprocs: [
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
	idx: #IDX
}
