#ENC_SRC_PATH:    "gs://zfish_unaligned/coarse_x0/encodings"
#ENC_MASKED_PATH: "gs://zfish_unaligned/coarse_x0/encodings_masked"

#DEFECTS_SRC_PATH: "gs://zfish_unaligned/coarse_x0/defect_mask"

#IMG_SRC_PATH:    "gs://zfish_unaligned/coarse_x0/raw_img"
#IMG_MASKED_PATH: "gs://zfish_unaligned/coarse_x0/raw_img_masked"

#XY_OUT_CHUNK: 1024

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2950]
	end_coord: [2048, 2048, 3100]
	resolution: [512, 512, 30]

}

#RESOLUTIONS: [
	[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],

]

#APPLY_MASK_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type": "apply_mask"
		"@mode": "partial"
	}
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0]]
	temp_layers_dirs: ["file://~.zutils/tmp_layers"]
	dst_resolution: _
	bbox:           #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	masks: [
		{
			"@type": "build_cv_layer"
			path:    #DEFECTS_SRC_PATH
			read_procs: [
			]
		},
	]
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
	}
	bbox: #BBOX
}

#APPLY_MASK_TO_ENCS_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in #RESOLUTIONS {
			#APPLY_MASK_TMPL & {
				dst_resolution: res
				src: path: #ENC_SRC_PATH
				dst: path: #ENC_MASKED_PATH
			}
		},
	]
}

#APPLY_MASK_TO_IMGS_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in #RESOLUTIONS {
			#APPLY_MASK_TMPL & {
				dst_resolution: res
				src: path: #IMG_SRC_PATH
				dst: path: #IMG_MASKED_PATH
			}
		},
	]
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x90"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:     300
batch_gap_sleep_sec: 1

local_test: false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				#APPLY_MASK_TO_IMGS_FLOW,
				#APPLY_MASK_TO_ENCS_FLOW,
			]

		},

	]
}
