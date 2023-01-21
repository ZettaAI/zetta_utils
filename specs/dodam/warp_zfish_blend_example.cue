#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/blending_example_rechunk13/"
#TEMP_PATH:  "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/blending_example_rechunk13/temp/"

#XY_OVERLAP:   1024
#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_blendable_apply_flow"
	operation: {
		"@type": "WarpOperation"
		mode:    "img"
	}
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [4096, 4096, 3000]
		end_coord: [8192, 8192, 3001]
		resolution: [32, 32, 30]
	}
	dst_resolution: [32, 32, 30]
	// processing_chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	// aggregation_chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	processing_chunk_size: [1056, 1040, 1]
	max_reduction_chunk_size: [2048, 2048, 1]
	// blend_pad: [#XY_OVERLAP, #XY_OVERLAP, 0]
	crop_pad: [64, 32, 0]
	blend_pad: [0, 0, 0]
	blend_mode:      "quadratic"
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
		on_info_exists:      "overwrite"
		write_preprocs: []
	}
	expand: true
}

"@type": "mazepa.execute"
target:
	#FLOW_TMPL
