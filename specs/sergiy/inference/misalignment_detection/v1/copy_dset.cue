#XY_OUT_CHUNK: 1024 * 3

#TGT_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x0"
#ZM1_PATH: "gs://sergiy_exp/aced/zfish/large_test_x7/base_enc_warped_-1"
#ZM2_PATH: "gs://sergiy_exp/aced/zfish/large_test_x7/base_enc_warped_-2"

#TGT_DST_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x1/tgt"
#ZM1_DST_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x1/src_zm1"
#ZM2_DST_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x1/src_zm2"

#STAGE_TMPL: {
	"@type": "build_write_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: [32, 32, 30]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3002]
		end_coord: [2048, 2048, 3015]
		resolution: [256, 256, 30]
	}
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #TGT_PATH
		on_info_exists:      "overwrite"
		info_chunk_size: [1024, 1024, 1]
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x10"
worker_resources: {
	memory: "18560Mi"
}

worker_replicas:     30
worker_lease_sec:    15
batch_gap_sleep_sec: 1.5

local_test: false

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#STAGE_TMPL & {
			src: path: #TGT_PATH
			dst: path: #TGT_DST_PATH
		},
		#STAGE_TMPL & {
			src: path: #ZM1_PATH
			dst: path: #ZM1_DST_PATH
		},
		#STAGE_TMPL & {
			src: path: #ZM2_PATH
			dst: path: #ZM2_DST_PATH
		},
	]
}
