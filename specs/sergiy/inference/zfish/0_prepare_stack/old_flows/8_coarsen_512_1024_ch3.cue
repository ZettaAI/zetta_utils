#SRC_PATH: "gs://zfish_unaligned/coarse_x0/tmp/coarsne_debug_x0_tmp_128nm_512nm_1.0_x0"
#DST_PATH: "gs://zfish_unaligned/coarse_x0/tmp/tmp_512nm_1024nm_1.0_ch3_x1"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/coarsener_gen_x1/tmp_512nm_1024nm_1.0_ch3_x1/last.ckpt.encoder.spec.json"

#CHUNK_SIZE: [1024, 1024, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [2048, 2048, 10]
	resolution: [512, 512, 30]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x14"
worker_replicas: 10
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: true

target: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "EncodingCoarsener"
			model_path: #MODEL_PATH
		}
		crop_pad: [128, 128, 0]
		res_change_mult: [2, 2, 1]
	}
	chunker: {
		"@type":      "VolumetricIndexChunker"
		"chunk_size": #CHUNK_SIZE
	}
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_chunk_size:     #DST_INFO_CHUNK_SIZE
		info_field_overrides: {
			data_type:    "int8"
			num_channels: 3
		}
		on_info_exists: "expect_same"
	}
	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BBOX
		resolution: [1024, 1024, 30]
	}
}
