#VERSION:    "v0"
#SRC_PATH:   "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/rigid_v2"
#FIELD_PATH: "gs://sergiy_exp/cns/alignment_tmp_\(#VERSION)/field"
#BCUBE: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2507]
	end_coord: [2048, 2048, 2509]
	resolution: [512, 512, 45]
}
#DST_PATH: "gs://sergiy_exp/cns/alignment_tmp_\(#VERSION)/warped"

#XY_CROP:      64
#XY_OUT_CHUNK: 2048
#DST_RESOLUTION: [256, 256, 45]
"@type": "mazepa.execute"
target: {
	"@type":        "build_warp_flow"
	mode:           "img"
	dst_resolution: #DST_RESOLUTION
	bbox:          #BCUBE
	crop: [#XY_CROP, #XY_CROP, 0]
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
	}
}
