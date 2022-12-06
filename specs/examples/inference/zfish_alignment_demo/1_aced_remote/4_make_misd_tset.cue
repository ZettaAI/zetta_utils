#VERSION:  "x2_masked"
#RIGIDITY: 50
#NUM_ITER: 100

#STAGE_PREFIX: "256_128_64_32nm"
#SRC1_PATH:    "gs://zfish_unaligned/coarse_x0/encodings_masked"
#DST1_PATH:    "gs://sergiy_exp/pairs_dsets/zfish_x0/src_enc"

#SRC2_PATH: "gs://sergiy_exp/pair_dset/zfish/alignment_\(#STAGE_PREFIX)/encodings_warped_to_z-1_\(#VERSION)_shift_x0"
#DST2_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x0/dst_enc"

#DST1_PATH:    ""
#XY_OUT_CHUNK: 2048

#STAGE_TMPL: {
	"@type": "build_write_flow"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: [64, 64, 30]
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

"@type": "mazepa.execute"
exec_queue: {
	"@type":            "mazepa.SQSExecutionQueue"
	name:               "aaa-zutils-x0"
	outcome_queue_name: "aaa-zutils-outcome-x0"
	pull_lease_sec:     60
}
target: [
	#STAGE_TMPL & {
		src: path: #SRC1_PATH
		dst: path: #DST1_PATH
	},
	#STAGE_TMPL & {
		src: path: #SRC1_PATH
		dst: path: #DST1_PATH
	},
]
