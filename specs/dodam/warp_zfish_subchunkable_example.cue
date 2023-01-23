#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/raw_masked"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/large_test_x8/afield_debug_x6"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend44/"
#TEMP_PATH1: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend1/temp/"
#TEMP_PATH0: "file:///tmp/zetta_cvols/"

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "lambda"
			lambda_str: "lambda src: src+1"
		}

	}

	start_coord: [4096, 4096, 3003]
	end_coord: [8192, 12288, 3005]
	coord_resolution: [32, 32, 30]

	dst_resolution: [32, 32, 30]

	// these are the args that need to be duplicated for all the levels
	// expand singletons, raise exception if lengths not same
	processing_chunk_sizes: [[4224, 4160, 1], [1056, 1056, 1]]
	processing_crop_pads: [[0, 32, 0], [0, 0, 0]]
	processing_blend_pads: [[0, 0, 0], [16, 16, 0]]
	processing_blend_modes: "quadratic"

	fov_crop_pad: [64, 64, 0]

	max_reduction_chunk_sizes: [4096, 4096, 1]

	temp_layers_dirs: [#TEMP_PATH1, #TEMP_PATH0]
	allow_cache_up_to_level: 1

	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
	}
}

"@type": "mazepa.execute"
target:
	#FLOW_TMPL
