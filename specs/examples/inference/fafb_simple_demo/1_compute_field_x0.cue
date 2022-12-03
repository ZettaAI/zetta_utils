#SRC_PATH: "gs://tmp_2w/inference_tests/raw_img_x20"
#DST_PATH: "gs://tmp_2w/inference_tests/field_x20"
#RIGIDITY: 100

"@type": "mazepa.execute"
target: {
	"@type": "build_compute_field_flow"
	chunk_size: [1024, 1024, 1]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 7, 1024 * 3, 2001]
		end_coord: [1024 * 9, 1024 * 5, 2005]
		resolution: [64, 64, 40]
	}
	dst_resolution: [64, 64, 40]

	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "align_with_online_finetunner"
			"@mode": "partial"
			sm:      #RIGIDITY
		}
		crop: [128, 128, 0]
	}
	tgt_offset: [0, 0, -1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_postprocs: [
			{
				"@type": "divide"
				"@mode": "partial"
				value:   255.0
			},
			{
				"@type": "add"
				"@mode": "partial"
				value:   -0.5
			},
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "expect_same"
	}
}
