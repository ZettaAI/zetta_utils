#SRC_PATH:    "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/match_offsets"
#MASKED_PATH: "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/match_offsets"

#MASK_SRC_PATH: "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/inverse_quality_debug_x0"

#XY_OUT_CHUNK: 1024

#APPLY_MASK_FLOW: {
	"@type": "build_apply_mask_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: [32, 32, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mask: {
		"@type": "build_cv_layer"
		path:    #MASK_SRC_PATH
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				value:   2
				mode:    ">="
			},
			{
				"@type": "filter_cc"
				"@mode": "partial"
				thr:     100
				mode:    "keep_large"
			},
			{
				"@type": "coarsen_mask"
				"@mode": "partial"
				width:   5
			},

		]

	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #MASKED_PATH
		info_reference_path: #SRC_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "expect_same"
	}
	bbox: #BBOX
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x37"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:     20
batch_gap_sleep_sec: 1

local_test: true

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		{
			"@type": "mazepa.concurrent_flow"
			stages: [
				#APPLY_MASK_FLOW,
			]

		},

	]

}
