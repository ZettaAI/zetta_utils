#PRECOARSE_ENC:     "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#PRECOARSE_IMG:     "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
#PRECOARSE_DEFECTS: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"

#FIELD: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"

#COARSE_ENC:     "gs://zfish_unaligned/coarse_x0/encodings"
#COARSE_IMG:     "gs://zfish_unaligned/coarse_x0/raw_img"
#COARSE_DEFECTS: "gs://zfish_unaligned/coarse_x0/defect_mask"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2950]
	end_coord: [2048, 2048, 3100]
	resolution: [512, 512, 30]
}

#WARP_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}

	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[512, 512, 0]]
	level_intermediaries_dirs: ["file://~.zutils/tmp_layers"]
	dst_resolution: _
	bbox:           #BBOX
	src: {
		"@type":    "build_cv_layer"
		path:       _
		read_procs: _ | *[]
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD
		data_resolution: [256, 256, 30]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		write_procs:    _ | *[]
	}
}

"@type":             "mazepa.execute_on_gcp_with_sqs"
worker_image:        "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x87"
worker_replicas:     100
batch_gap_sleep_sec: 0.2
local_test:          false

worker_resources: {
	memory: "18560Mi"
}
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in [[32, 32, 30], [64, 64, 30], [128, 128, 30]] {
			#WARP_TMPL & {
				dst_resolution: res
				src: path: #PRECOARSE_IMG
				dst: path: #COARSE_IMG
				op: mode:  "img"
				//processing_chunk_sizes: [[4096, 4096, 1], [1024, 1024, 1]]
			}
		},
		for res in [[256, 256, 30], [512, 512, 30]] {
			#WARP_TMPL & {
				dst_resolution: res
				src: path: #PRECOARSE_IMG
				dst: path: #COARSE_IMG
				op: mode:  "img"
				//processing_chunk_sizes: [[1024, 1024, 1], [1024, 1024, 1]]
			}
		},
		for res in [[32, 32, 30], [64, 64, 30], [128, 128, 30]] {
			#WARP_TMPL & {
				dst_resolution: res
				src: path: #PRECOARSE_ENC
				dst: path: #COARSE_ENC
				op: mode:  "img"
				//processing_chunk_sizes: [[4096, 4096, 1], [1024, 1024, 1]]
			}
		},
		for res in [[256, 256, 30], [512, 512, 30]] {
			#WARP_TMPL & {
				dst_resolution: res
				src: path: #PRECOARSE_ENC
				dst: path: #COARSE_ENC
				op: mode:  "img"
				//processing_chunk_sizes: [[1024, 1024, 1], [1024, 1024, 1]]
			}
		},

		for res in [[32, 32, 30], [64, 64, 30], [128, 128, 30]] {
			#WARP_TMPL & {
				dst_resolution: res
				src: path: #PRECOARSE_DEFECTS
				dst: path: #COARSE_DEFECTS
				op: mode:  "mask"
				//processing_chunk_sizes: [[4096, 4096, 1], [1024, 1024, 1]]
				src: read_procs: [
					{
						"@type":    "binary_closing"
						"@mode":    "partial"
						iterations: 2 * 512 / res[0]
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
				]

				dst: write_procs: [
					{
						"@type":    "binary_closing"
						"@mode":    "partial"
						iterations: 2 * 512 / res[0]
					},
					{
						"@type": "filter_cc"
						"@mode": "partial"
						thr:     15
						mode:    "keep_large"
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			}
		},
		for res in [[256, 256, 30], [512, 512, 30]] {
			#WARP_TMPL & {
				dst_resolution: res
				src: path: #PRECOARSE_DEFECTS
				dst: path: #COARSE_DEFECTS
				op: mode:  "mask"
				//processing_chunk_sizes: [[1024, 1024, 1], [1024, 1024, 1]]
				src: read_procs: [
					{
						"@type":    "binary_closing"
						"@mode":    "partial"
						iterations: 2 * 512 / res[0]
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
				]

				dst: write_procs: [
					{
						"@type":    "binary_closing"
						"@mode":    "partial"
						iterations: 2 * 512 / res[0]
					},
					{
						"@type": "filter_cc"
						"@mode": "partial"
						thr:     15
						mode:    "keep_large"
					},
					{
						"@type": "to_uint8"
						"@mode": "partial"
					},
				]
			}
		},

	]
}
