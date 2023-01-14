#SRC_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_encodings"
#FIELD_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/coarse/v3/field/composed_drift_corrected"
#DST_PATH:   "gs://zfish_unaligned/coarse_x0/encodings"

#XY_OVERLAP:   512
#XY_OUT_CHUNK: 2048 + 1024

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 20]
	end_coord: [1024, 1024, 200]
	resolution: [512, 512, 30]
}

#FLOW_TMPL: {
	"@type": "build_warp_flow"
	crop: [256, 256, 0]
	mode: "img"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
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
		info_chunk_size: [1024, 1024, 1]
	}
	bcube:          #BCUBE
	dst_resolution: _
}

#RESOLUTIONS: [
	//[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 2
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x20"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:     100
worker_lease_sec:    10
batch_gap_sleep_sec: 3

local_test: false

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for res in #RESOLUTIONS {
			#FLOW_TMPL & {
				dst_resolution: res
			}
		},
	]
}
