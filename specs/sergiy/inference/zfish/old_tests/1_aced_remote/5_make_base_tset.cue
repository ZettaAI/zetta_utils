#VERSION:  "x2_masked"
#RIGIDITY: 50
#NUM_ITER: 100

#STAGE_PREFIX: "256_128_64_32nm"
#SRC1_PATH:    "gs://zfish_unaligned/coarse_x0/raw_img"
#DST1_PATH:    "gs://sergiy_exp/pairs_dsets/zfish_x0/src"

#SRC2_PATH: "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/raw_warped_to_z-1_\(#VERSION)_shift_x0"
#DST2_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x0/dst"

#XY_OUT_CHUNK: 2048

#STAGE_TMPL: {
	"@type": "build_write_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: [32, 32, 30]
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3000]
		end_coord: [2048, 2048, 3020]
		resolution: [256, 256, 30]
	}
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: #SRC1_PATH
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

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:inference_x11"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     30
worker_lease_sec:    15
batch_gap_sleep_sec: 1.5
local_test:          false
target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		#STAGE_TMPL & {
			src: path: #SRC1_PATH
			dst: path: #DST1_PATH
		},
		#STAGE_TMPL & {
			src: path: #SRC2_PATH
			dst: path: #DST2_PATH
		},
	]
}
