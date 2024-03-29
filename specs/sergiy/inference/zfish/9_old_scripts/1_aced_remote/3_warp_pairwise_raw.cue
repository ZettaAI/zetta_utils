#VERSION:  "x2_masked"
#RIGIDITY: 50
#NUM_ITER: 100

#STAGE_PREFIX: "256_128_64_32nm"
#SRC_PATH:     "gs://zfish_unaligned/coarse_x0/raw_img"
#FIELD_PATH:   "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/field_\(#VERSION)"
#DST_PATH:     "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE_PREFIX)/raw_warped_to_z-1_\(#VERSION)_shift_x0"

#XY_CROP:      512
#XY_OUT_CHUNK: 2048

#STAGE_TMPL: {
	"@type": "build_warp_flow"
	mode:    "img"
	chunk_size: [#XY_OUT_CHUNK, #XY_OUT_CHUNK, 1]
	dst_resolution: _
	bbox: {
		"@type": "BBox3D.from_coords"
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
		index_procs: [
			{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, -1]
				resolution: [4, 4, 30]
			},
		]
	}
}

#RESOLUTIONS: [
	//32,
	//64,
	128,
	256,
]
"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:inference_x11"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     30
worker_lease_sec:    30
batch_gap_sleep_sec: 0.5
local_test:          false

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		for res in #RESOLUTIONS {
			#STAGE_TMPL & {
				dst_resolution: [res, res, 30]
			}
		},
	]
}
