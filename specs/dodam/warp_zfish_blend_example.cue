#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/blending_example_quad/"
#TEMP_PATH:  "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/blending_example_quad/temp/"

#XY_OVERLAP:   512
#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_blendable_apply_flow"
	operation: {
		"@type": "WarpOperation"
		mode:    "img"
	}
	bcube: {
		"@type": "BoundingCube"
		start_coord: [256, 256, 3000]
		end_coord: [512, 512, 3003]
		resolution: [512, 512, 30]
	}
	dst_resolution: [32, 32, 30]
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	blend_pad: [#XY_OVERLAP, #XY_OVERLAP, 0]
	blend_mode:      "linear"
	temp_layers_dir: #TEMP_PATH

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
