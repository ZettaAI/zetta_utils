#SRC_PATH:  "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#DST_PATH:  "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/copy_dst0"
#TEMP_PATH: "file:///tmp/zetta_cvols"

#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_blendable_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":      "lambda"
			"lambda_str": "lambda src: src"
		}
		crop_pad: [1, 1, 0]
	}
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [8192, 8192, 3000]
		end_coord: [16384, 16384, 3010]
		resolution: [16, 16, 30]
	}
	dst_resolution: [16, 16, 30]
	processing_chunk_size: [2048, 2048, 1]
	max_reduction_chunk_size: [2048, 2048, 1]
	crop_pad: [0, 0, 0]
	blend_pad: [0, 0, 0]
	blend_mode:      "quadratic"
	temp_layers_dir: #TEMP_PATH
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_postprocs: []
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
		write_preprocs: []
	}
}

"@type": "mazepa.execute"
target:
	#FLOW_TMPL
