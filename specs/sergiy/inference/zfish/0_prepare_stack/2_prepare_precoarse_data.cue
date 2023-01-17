#ENC_SRC_PATH:    "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#ENC_MASKED_PATH: "gs://zfish_unaligned/precoarse_x0/encodings_masked"

#DEFECTS_SRC_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"

#IMG_SRC_PATH:    "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
#IMG_MASKED_PATH: "gs://zfish_unaligned/precoarse_x0/raw_masked"

#XY_OUT_CHUNK: 2048

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 0]
	end_coord: [1024, 1024, 50]
	resolution: [512, 512, 30]
}

#RESOLUTIONS: [
	//[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

#APPLY_MASK_TMPL: {
	"@type": "build_apply_mask_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: _
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	mask: {
		"@type": "build_cv_layer"
		path:    #DEFECTS_SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: _
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "expect_same"
	}
	bcube: #BCUBE
}

#APPLY_MASK_TO_ENCS_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in #RESOLUTIONS {
			#APPLY_MASK_TMPL & {
				dst_resolution: res
				src: path:                #ENC_SRC_PATH
				dst: path:                #ENC_MASKED_PATH
				dst: info_reference_path: #ENC_SRC_PATH
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
				src: path:                #IMG_SRC_PATH
				dst: path:                #IMG_MASKED_PATH
				dst: info_reference_path: #IMG_SRC_PATH
			}
		},
	]
}
"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x28"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:     20
worker_lease_sec:    10
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
