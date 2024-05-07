#SRC_PATH: "assets/inputs/fafb_v15_img_128_128_40-2048-3072_2000-2050_uint8"
#DST_PATH: "assets/outputs/test_uint8_exc_no_dst_but_max_reduction_chunk_size"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [64 * 1024, 64 * 1024, 2000]
	end_coord: [96 * 1024, 96 * 1024, 2005]
	resolution: [4, 4, 40]
}

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: print(src)"
	}
	processing_chunk_sizes: [[1024, 1024, 1], [512, 512, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	max_reduction_chunk_size: [1024, 1024, 1]
	skip_intermediaries:    true
	expand_bbox_processing: true
	dst_resolution: [128, 128, 40]
	bbox: #BBOX
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}
	dst: null
}

"@type": "mazepa.execute"
target:  #FLOW