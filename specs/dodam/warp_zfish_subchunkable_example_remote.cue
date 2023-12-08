#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/raw_masked"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/large_test_x8/afield_debug_x6"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend50/"
#TEMP_PATH2: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend1/temp/"
#TEMP_PATH1: "file:///tmp/ramdisk/zetta_cvols/"
#TEMP_PATH0: "file:///tmp/zetta_cvols/"

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
	}

	start_coord: [4096 * 3, 4096 * 4, 3003]
	end_coord: [12288 * 2, 12288 * 2, 3011]
	coord_resolution: [16, 16, 30]

	dst_resolution: [32, 32, 30]

	// these are the args that need to be duplicated for all the levels
	// expand singletons, raise exception if lengths not same
	//processing_chunk_sizes: [[8192, 8192, 1], [2048, 2048, 1], [512, 512, 1]]
	//processing_crop_pads: [[0, 0, 0], [16, 16, 0], [32, 32, 0]]
	//processing_blend_pads: [[0, 0, 0], [0, 0, 0], [16, 16, 0]]
	//level_intermediaries_dirs: [#TEMP_PATH2, #TEMP_PATH1, #TEMP_PATH0]
	processing_chunk_sizes: [[8192, 8192, 1], [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	processing_blend_pads: [[0, 0, 0], [16, 16, 0]]
	processing_blend_modes: "quadratic"
	level_intermediaries_dirs: [#TEMP_PATH2, #TEMP_PATH1]

	max_reduction_chunk_sizes: [4096, 4096, 1]
	expand_bbox_resolution:  true
	expand_bbox_processing:  true
	shrink_processing_chunk: false

	auto_divisibility: true

	skip_intermediaries: false
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
		field: {
			"@type": "build_cv_layer"
			path:    #FIELD_PATH
			data_resolution: [32, 32, 30]
			interpolation_mode: "field"
		}
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
	}

}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:dodam-dev"
worker_replicas: 4

worker_resources: {
	"nvidia.com/gpu": "1"
}

num_procs: 4
semaphores_spec: {
	read:  4
	write: 4
	cuda:  1
	cpu:   4
}
target:     #FLOW_TMPL
debug:      false
local_test: false
