#RIGIDITY:     40
#NUM_ITER:     150
#VERSION:      "test_rig\(#RIGIDITY)_iter\(#NUM_ITER)"
#STAGE_PREFIX: "256_128_64_32nm"
#SRC_PATH:     "gs://zfish_unaligned/coarse_x0/encodings_masked"
#DST_PATH:     "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
#RESOLUTIONS: [
	[256, 256, 30],
	[128, 128, 30],
	[64, 64, 30],
	[32, 32, 30],
]

#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":  "align_with_online_finetuner"
			"@mode":  "partial"
			sm:       #RIGIDITY
			num_iter: #NUM_ITER
		}
		crop: [128, 128, 0]
	}
	chunk_size: [2048, 2048, 1]
}

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 1000
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x0"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:     10
worker_lease_sec:    40
batch_gap_sleep_sec: 5

local_test: false

target: {
	"@type": "build_compute_field_multistage_flow"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3001]
		end_coord: [2048, 2048, 3016]
		resolution: [256, 256, 30]
	}
	stages: [
		for res in #RESOLUTIONS {
			#STAGE_TMPL & {'dst_resolution': res}
		},
	]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	tgt_offset: [0, 0, -1]
	tgt: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "expect_same"
	}
	tmp_layer_dir: "\(#DST_PATH)/tmp"
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "expect_same"
	}
}
