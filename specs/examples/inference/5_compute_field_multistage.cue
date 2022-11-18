#SRC_PATH: "gs://tmp_2w/inference_tests/raw_img_x0"
#DST_PATH: "gs://tmp_2w/inference_tests/field_x0"

#XY_OVERLAP:   512
#XY_OUT_CHUNK: 1024
#RIGIDITY:     100

#IDX: {
	"@type": "VolumetricIndex"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [-#XY_OVERLAP/2 + 1024*7, -#XY_OVERLAP/2 + 1024*3, 2001]
		end_coord: [#XY_OVERLAP + 1024*9, #XY_OVERLAP + 1024*5, 2005]
		resolution: [64, 64, 40]
	}
	resolution: [64, 64, 40]
}

"@type": "mazepa_execute"
target: {
	"@type":           "compute_z_neighbor_fields"
	farthest_neighbor: 1
	compute_field_method: {
		"@type": "SimpleVolumetricTaskFactory"
		fn: {
			"@type": "align_with_online_finetunner"
			"@mode": "partial"
			sm:      #RIGIDITY
		}
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
	dst_dir: #DST_PATH
	dst_layer_builder: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "expect_same"
	}
	idx: #IDX
}
