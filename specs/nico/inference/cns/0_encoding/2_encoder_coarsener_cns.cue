#SRC_PATH "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.0002_post1.8_cns"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/M3_M8_enc"

#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_encoder_coarsener/tmp_M3_M8_conv2_lr0.0001_post1.2_cns_all_newenc/last.ckpt.model.spec.json"

#CHUNK_SIZE: [1024, 1024, 1]
#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [10240, 0, 3000]
	end_coord: [22528, 28672, 3002]
	resolution: [32, 32, 45]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_unet_pow"
worker_replicas: 10
worker_resources: {
	"nvidia.com/gpu": "1"
}
local_test: true

target: {
	"@type": "build_subchunkable_apply_flow"
    bbox: #BBOX

	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			data_type: "int8"
		}
		info_chunk_size: #DST_INFO_CHUNK_SIZE
		on_info_exists:  "overwrite"
	}
    dst_resolution: [1024, 1024, 45]

	// these are the args that need to be duplicated for all the levels
	// expand singletons, raise exception if lengths not same
	processing_chunk_sizes: [[32768, 32768, 1], [1024, 1024, 1]]
	max_reduction_chunk_sizes: [4096, 4096, 1]
	fn_or_op_crop_pad: [128, 128, 0]
	crop_pads: [[0, 0, 0], [0, 0, 0]]
	blend_pads: [[0, 0, 0], [0, 0, 0]]
	blend_modes: "linear"
	temp_layers_dirs: [#TEMP_PATH, #TEMP_PATH, #TEMP_PATH]


	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseEncoder"
			model_path: #MODEL_PATH
		}
		crop_pad: [64, 64, 0]
	}
	chunker: {
		"@type":      "VolumetricIndexChunker"
		"chunk_size": #CHUNK_SIZE
	}
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}


	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BBOX
		resolution: [32, 32, 45]
	}
}