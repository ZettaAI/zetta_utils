#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#DST_PATH:   "gs://sergiy_exp/aced/zfish/coarse_v1_encodings"

#XY_OVERLAP:   512
#XY_OUT_CHUNK: 2048

#IDX_TMPL: {
	"@type":    "VolumetricIndex"
	resolution: _
	bcube:
	{
		"@type": "pad_bcube"
		pad: [#XY_OVERLAP / 2, #XY_OVERLAP / 2, 0]
		pad_resolution: resolution // matches the idx resolution
		bcube: {
			"@type": "BoundingCube"
			start_coord: [0, 0, 3000]
			end_coord: [1024, 1024, 3100]
			resolution: [512, 512, 30]
		}
	}
}

#FLOW_TMPL: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "WarpOperation"
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
	idx: _
}

#RESOLUTIONS: [
	//[512, 512, 30],
	//[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

"@type": "mazepa.execute"
target: [
	for res in #RESOLUTIONS {
		#FLOW_TMPL & {
			idx: #IDX_TMPL & {'resolution': res}
		}
	},
]
