#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/subchunks_0/"
#TEMP_PATH:  "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/subchunks_0/temp/"

#XY_OVERLAP:   512
#XY_CROP:      256
#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
	}
	bcube: {
		"@type": "BoundingCube"
		start_coord: [256, 256, 3000]
		end_coord: [1024, 1024, 3001]
		resolution: [512, 512, 30]
	}
	dst_resolution: [32, 32, 30]

	// these are the args that need to be duplicated for all the levels
	// expand singletons, raise exception if lengths not same
	processing_chunk_sizes: [[6144, 6144, 1], [3104, 3088, 1], [1552, 1544, 1]]
	max_reduction_chunk_sizes: [4096, 4096, 1]
	fn_or_op_crop_pad: [16, 16, 0]
	crop_pads: [[0, 0, 0], [32, 16, 0], [0, 0, 0]]
	blend_pads: [[0, 0, 0], [0, 0, 0], [32, 32, 0]]
	blend_modes: "linear"
	temp_layers_dirs: [#TEMP_PATH, #TEMP_PATH, #TEMP_PATH]
	// TODO add execution mode / resources
	// gs -> file -> mem

	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_postprocs: []
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
		data_resolution: [256, 256, 30]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "expect_same"
		write_preprocs: []
	}
}

"@type": "mazepa.execute"
target:
	#FLOW_TMPL
