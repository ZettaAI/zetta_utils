#SRC_PATH:  "gs://zfish_unaligned/coarse_x0/raw_img"
#DST_PATH:  "gs://zfish_unaligned/coarse_x0/raw_masked"
#MASK_PATH: "gs://zfish_unaligned/coarse_x0/defect_mask"

#RESOLUTIONS: [
	[512, 512, 30],
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

#CHUNK_SIZE: 2048
#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 3000]
	end_coord: [2048, 2048, 3020]
	resolution: [512, 512, 30]
}

#FLOW_TMPL: {
	"@type": "build_apply_mask_flow"
	chunk_size: [#CHUNK_SIZE, #CHUNK_SIZE, 1]
	dst_resolution: _
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mask: {
		"@type": "build_cv_layer"
		path:    #MASK_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
	}
	bcube: #BCUBE
}

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 5
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x2"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:     30
worker_lease_sec:    10
batch_gap_sleep_sec: 5

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
