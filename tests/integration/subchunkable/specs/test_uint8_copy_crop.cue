#SRC_PATH: "tests/integration/assets/inputs/fafb_v15_img_128_128_40-2048-3072_2000-2050"
#DST_PATH: "tests/integration/assets/outputs/test_uint8_copy_crop"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [64 * 1024, 64 * 1024, 2000]
	end_coord: [96 * 1024, 96 * 1024, 2050]
	resolution: [4, 4, 40]
}

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[1024, 1024, 1], [512, 512, 1]]
	processing_crop_pads: [[0, 0, 0], [64, 64, 0]]
	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	max_reduction_chunk_sizes: [[1024, 1024, 1], [1024, 1024, 1]]
	level_intermediaries_dirs: ["tests/integration/assets/temp/", "tests/integration/assets/temp/"]
	expand_bbox: true
	dst_resolution: [128, 128, 40]
	bbox: #BBOX
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
	}
}

"@type": "mazepa.execute"
target:  #FLOW
