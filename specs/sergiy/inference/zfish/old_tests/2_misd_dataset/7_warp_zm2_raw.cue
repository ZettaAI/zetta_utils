#VERSION: "test_rig40_iter150_zm2"

#STAGE_PREFIX: "256_128_64_32nm"
#SRC_PATH:     "gs://zfish_unaligned/coarse_x0/raw_img"
#FIELD_PATH:   "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
#DST_PATH:     "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/raw_warped_to_z-2_\(#VERSION)_shift_x0"

#XY_CROP:      512
#XY_OUT_CHUNK: 2048

#STAGE_TMPL: {
	"@type": "build_warp_flow"
	mode:    "img"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: _
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3000]
		end_coord: [2048, 2048, 3020]
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
				offset: [0, 0, -2]
				resolution: [4, 4, 30]
			},
		]
	}
}

#RESOLUTIONS: [
	32,
	//64,
	//128,
	//256,
]
"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x2"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     30
worker_lease_sec:    30
batch_gap_sleep_sec: 0.5
local_test:          false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		for res in #RESOLUTIONS {
			#STAGE_TMPL & {
				dst_resolution: [res, res, 30]
			}
		},
	]
}
