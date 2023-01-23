#SRC_PATH:   "gs://zfish_unaligned/coarse_x0/raw_masked"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/large_test_x8/afield_debug_x6"
#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend1/"
#TEMP_PATH:  "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/newblend1/temp/"

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:dodam_subchunkable_x3"
worker_replicas: 16

worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "build_blendable_apply_flow"
	operation: {
		"@type": "WarpOperation"
		mode:    "img"
		crop_pad: [1, 1, 0]
	}
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [4096, 4096, 3003]
		end_coord: [8192, 12288, 3011]
		resolution: [32, 32, 30]
	}
	dst_resolution: [32, 32, 30]
	processing_chunk_size: [1024, 1024, 1]
	max_reduction_chunk_size: [4096, 4096, 1]
	crop_pad: [0, 0, 0]
	blend_pad: [0, 0, 0]
	blend_mode:      "quadratic"
	temp_layers_dir: #TEMP_PATH
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
