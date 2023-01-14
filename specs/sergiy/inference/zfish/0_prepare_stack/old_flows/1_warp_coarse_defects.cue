#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"
#DST_PATH:   "gs://zfish_unaligned/coarse_x0/defect_mask"

#XY_CROP:      256
#XY_OUT_CHUNK: 2048

#FLOW_TMPL: {
	"@type": "build_warp_flow"
	crop: [#XY_CROP, #XY_CROP, 0]
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	mode:           "mask"
	mask_value_thr: 0.1
	dst_resolution: _
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_postprocs: [
			{
				"@type": "rearrange"
				"@mode": "partial"
				pattern: "c x y z -> z c x y"
			},
			{
				"@type":    "binary_closing"
				"@mode":    "partial"
				iterations: 2 * 512 / dst_resolution[0]
			},
			{
				"@type": "filter_cc"
				"@mode": "partial"
				thr:     15
				mode:    "keep_large"
			},
			{
				"@type": "coarsen_mask"
				"@mode": "partial"
				width:   1
			},
			{
				"@type": "rearrange"
				"@mode": "partial"
				pattern: "z c x y -> c x y z"
			},
		]
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
		write_preprocs: [
			{
				"@type": "rearrange"
				"@mode": "partial"
				pattern: "c x y z -> z c x y"
			},
			{
				"@type":    "binary_closing"
				"@mode":    "partial"
				iterations: 2 * 512 / dst_resolution[0]
			},
			{
				"@type": "filter_cc"
				"@mode": "partial"
				thr:     15
				mode:    "keep_large"
			},
			{
				"@type": "rearrange"
				"@mode": "partial"
				pattern: "z c x y -> c x y z"
			},
		]
	}
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3010]
		end_coord: [2048, 2048, 3020]
		resolution: [512, 512, 30]
	}
}

#RESOLUTIONS: [
	[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

"@type": "mazepa.execute"
target: [
	for res in #RESOLUTIONS {
		#FLOW_TMPL & {dst_resolution: res}
	},
]
