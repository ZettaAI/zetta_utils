#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/raw_masked"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/large_test_x8/afield_debug_x6"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend45_blend_l4/"
#TEMP_PATH:  "file:///tmp/zetta_cvols/"

#FLOW_TMPL: {
	"@type": "build_blendable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
	}
	start_coord: [4096, 4096, 3003]
	end_coord: [8192, 8192, 3004]
	coord_resolution: [32, 32, 30]

	fov_crop_pad: [0, 0, 0]

	dst_resolution: [32, 32, 30]

	processing_crop_pad: [8, 8, 0]
	processing_chunk_size: [1024, 1024, 1]
	processing_blend_pad: [16, 16, 0]
	processing_blend_mode: "linear"

	max_reduction_chunk_size: [4096, 4096, 1]

	temp_layers_dir: #TEMP_PATH
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_procs: []
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
		data_resolution: [32, 32, 30]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
		write_procs: []
	}
}

"@type": "mazepa.execute"
target:
	#FLOW_TMPL
