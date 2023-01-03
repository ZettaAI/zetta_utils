#VERSION:  "x2_masked"
#RIGIDITY: 50
#NUM_ITER: 100

#STAGE_PREFIX: "256_128_64_32nm"
#FIELD_PATH:   "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"

#XY_OUT_CHUNK: 2048

#STAGE_TMPL: {
	"@type": "build_interpolate_flow"
	mode:    "field"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	src_resolution: _
	dst_resolution: _
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3000]
		end_coord: [2048, 2048, 3020]
		resolution: [256, 256, 30]
	}
	src: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
	}
}

#RESOLUTIONS: [
	32, 64, 128,
]
"@type": "mazepa.execute"
exec_queue: {
	"@type":            "mazepa.SQSExecutionQueue"
	name:               "aaa-zutils-x0"
	outcome_queue_name: "aaa-zutils-outcome-x0"
	pull_lease_sec:     6
}
target: {
	"@type": "mazepa.seq_flow"
	stages: [
		for res in #RESOLUTIONS {
			#STAGE_TMPL & {
				src_resolution: [res, res, 30]
				dst_resolution: [2 * res, 2 * res, 30]
			}
		},
	]
}
