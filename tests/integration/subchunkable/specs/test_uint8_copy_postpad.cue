#SRC_PATH: "assets/inputs/fafb_v15_img_128_128_40-2048-3072_2000-2050_uint8"
#DST_PATH: "assets/outputs/test_uint8_copy_postpad"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [64 * 1024, 64 * 1024, 2000]
	end_coord: [96 * 1024, 96 * 1024, 2001]
	resolution: [4, 4, 40]
}

#FLOW: {
	"@type": "build_postpad_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_input_sizes: [[820, 938, 1], [234, 192, 1]]
	processing_crop: [3, 5, 0]
	processing_blend: [7, 11, 0]
	max_reduction_chunk_size: [1024, 1024, 1]
	level_intermediaries_dirs: ["assets/temp/", "assets/temp/"]
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
