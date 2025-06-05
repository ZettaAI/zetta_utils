"@version": "0.4"
#SRC_PATH: "assets/inputs/fafb_v15_img_128_128_40-2048-3072_2000-2050_uint8"
#DST_PATH: "assets/outputs/test_uint8_copy_info_bbox_oddsize"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [64 * 1024-66, 64 * 1024+67, 2000]
	end_coord: [96 * 1024-33, 96 * 1024-67, 2005]
	resolution: [4, 4, 40]
}

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[1024, 1024, 1]]
	processing_crop_pads: [[0, 0, 0]]
	level_intermediaries_dirs: ["assets/temp/"]
	expand_bbox_resolution: true
	expand_bbox_processing: true
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
		info_bbox:           #BBOX
		info_scales:         [[128, 128, 40]]
		info_inherit_all_params:  true
	}
}

"@type": "mazepa.execute"
target:  #FLOW
