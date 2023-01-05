#RIGIDITY: 40
#NUM_ITER: 150

#VERSION: "test_rig\(#RIGIDITY)_iter\(#NUM_ITER)"

#STAGE_PREFIX: "256_128_64_32nm"

//#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/encodings_masked"
//#FIELD_PATH: "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
//#DST_PATH:   "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/encodings_warped_to_z-1_\(#VERSION)_shift_x0"

#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/raw_img"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
#DST_PATH:   "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/raw_warped_to_z-1_\(#VERSION)_shift_x0"

#XY_CROP:      512
#XY_OUT_CHUNK: 2048

"@type":        "mazepa.execute_on_gcp_with_sqs"
max_task_retry: 5
worker_image:   "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x0"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas:     25
worker_lease_sec:    10
batch_gap_sleep_sec: 5

local_test: false

target: {
	"@type": "build_warp_flow"
	mode:    "img"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: [32, 32, 30]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3001]
		end_coord: [2048, 2048, 3016]
		resolution: [256, 256, 30]
	}
	crop: [#XY_CROP, #XY_CROP, 0]
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
		info_chunk_size: [1024, 1024, 1]
		index_adjs: [
			{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, -1]
				resolution: [4, 4, 30]
			},
		]
	}
}
