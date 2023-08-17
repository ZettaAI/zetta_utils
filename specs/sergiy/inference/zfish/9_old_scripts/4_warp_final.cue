#PRECOARSE_IMG: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"

#COARSE_FIELD: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#FINE_FIELD:   "gs://sergiy_exp/aced/zfish/cutout_g_x9/afield_try_x0"

#COMBINED_FIELD: "gs://sergiy_exp/aced/zfish/cutout_g_x9/afield_combined_try_x0"
#FINAL_IMG:      "gs://sergiy_exp/aced/zfish/cutout_g_x9/final_img_try_x0"

#BBOX: {
	"@type": "BBox3D.from_coords"
	//start_coord: [0, 0, 2958]
	start_coord: [24 * 1024, 32 * 1024, 2958]
	end_coord: [40 * 1024, 48 * 1024, 3093]
	resolution: [4, 4, 30]
}

#DOWNS_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2958]
	end_coord: [40 * 1024, 48 * 1024, 3093]
	resolution: [4, 4, 30]
}
#WARP_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}

	processing_chunk_sizes: [[1024, 1024, 1]]
	processing_crop_pads: [[512, 512, 0]]
	level_intermediaries_dirs: ["file://~.zutils/tmp_layers"]
	dst_resolution: _
	bbox:           #BBOX
	src: {
		"@type":            "build_cv_layer"
		path:               _
		data_resolution:    _ | *null
		interpolation_mode: _ | *null
	}
	field: {
		"@type":            "build_cv_layer"
		path:               _
		data_resolution:    _ | *null
		interpolation_mode: "field"
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _

		info_reference_path: src.path

		info_chunk_size: [1024, 1024, 1]

		on_info_exists: "overwrite"
	}

}

#DOWNS_TMPL: {

	"@type":        "build_interpolate_flow"
	mode:           "img"
	src_resolution: _
	dst_resolution: _

	chunk_size: [2048, 2048, 1]

	bbox: #DOWNS_BBOX
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
}

"@type":             "mazepa.execute_on_gcp_with_sqs"
worker_image:        "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x90"
worker_replicas:     30
batch_gap_sleep_sec: 0.2
local_test:          false

worker_resources: {
	memory: "18560Mi"
}
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		#WARP_TMPL & {
			src: path: #COARSE_FIELD
			src: data_resolution: [256, 256, 30]
			src: interpolation_mode: "field"

			field: path: #FINE_FIELD

			dst: path: #COMBINED_FIELD
			dst_resolution: [32, 32, 30]
			op: mode: "field"
		},
		#WARP_TMPL & {
			dst_resolution: [16, 16, 30]
			src: path:   #PRECOARSE_IMG
			field: path: #COMBINED_FIELD
			field: data_resolution: [32, 32, 30]
			dst: path: #FINAL_IMG
			op: mode:  "img"
		},
		for res in [16, 32, 64] {
			#DOWNS_TMPL & {
				src: path: #FINAL_IMG
				src_resolution: [res, res, 30]
				dst_resolution: [res * 2, res * 2, 30]
			}
		},
	]
}
