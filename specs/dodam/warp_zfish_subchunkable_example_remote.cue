#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/raw_masked"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/large_test_x8/afield_debug_x6"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend1/"
#TEMP_PATH1: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend1/temp/"
#TEMP_PATH0: "file:///opt/zetta_utils/"

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:dodam_subchunkable_x15"
worker_replicas: 16

worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "img"
	}
	start_coord: [4096, 4096, 3003]
	end_coord: [8192, 12288, 3008]
	coord_resolution: [32, 32, 30]

	dst_resolution: [32, 32, 30]

	processing_chunk_sizes: [[4096, 4096, 1], [1024, 1024, 1]]
	max_reduction_chunk_sizes: [4096, 4096, 1]

	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	processing_blend_pads: [[0, 0, 0], [0, 0, 0]]
	processing_blend_modes: "quadratic"

	fov_crop_pad: [0, 0, 0]

	temp_layers_dirs: [#TEMP_PATH1, #TEMP_PATH0]
	allow_cache_up_to_level: 1
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_procs: []
	}
	field: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
		data_resolution: [32, 32, 30]
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		on_info_exists:      "overwrite"
		write_procs: []
	}
}
