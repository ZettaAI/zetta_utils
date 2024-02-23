#SRC_PATH: "assets/inputs/fafb_v15_img_128_128_40-2048-3072_2000-2050_float32"
#DST_PATH: "assets/outputs/test_float32_copy_defer"

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
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[1024, 1024, 1]]
	processing_blend_pads: [[64, 64, 0]]
	processing_crop_pads: [[0, 0, 0]]
    processing_blend_modes: "defer"
	max_reduction_chunk_sizes: [[1024, 1024, 1]]
	// Normally you would specify a different location for the intermediate
	// directory, but for the sake of integration testing, we'll use the
	// output directory, so the test harness can find the intermediate files.
	level_intermediaries_dirs: [#DST_PATH]
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
		write_procs: [
			{
				"@type":    "lambda"
				lambda_str: "lambda data: data + 1"
			}]
	}
}

"@type": "mazepa.execute"
target:  #FLOW
