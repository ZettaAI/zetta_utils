#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#DST_PATH:   "gs://zfish_unaligned/coarse_x0/raw_img"

#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_warp_flow"
	mode:    "img"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	crop: [256, 256, 0]
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [0, 0, 3000]
		end_coord: [2048, 2048, 3020]
		resolution: [512, 512, 30]
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
	dst_resolution: _
}

#RESOLUTIONS: [
	//[512, 512, 30],
	//[256, 256, 30],
	//[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

"@type": "mazepa.execute"
target: [
	for res in #RESOLUTIONS {
		#FLOW_TMPL & {dst_resolution: res}
	},
]
