// #SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.0002_post1.8_cns"
#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/M3_M8_img_conv2_post1.03_fmt1.1"

// #MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_encoder_coarsener/tmp_M3_M8_conv2_lr0.0001_post1.1_cns_all_newenc/last.ckpt.model.spec.json"
#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_encoder_coarsener/M3_M8_conv2_lr0.0001_post1.03_fmt1.1_cns_all_rawimg/last.ckpt.model.spec.json"

#CHUNK_SIZE: [1024, 1024, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#RESOLUTION: [1024, 1024, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3201]
	end_coord: [32768, 32768, 3202]
	resolution: [32, 32, 45]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230207_m6_inference"
worker_replicas:  10
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: true

target: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseEncoderCoarsenerChunked"
			model_path: #MODEL_PATH
			tile_pad_in: 128
			tile_size: 1024
			ds_factor: 32
		}
		crop_pad: [128, 128, 0]
		res_change_mult: [32, 32, 1]
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
			data_type: "int8"
		}
		on_info_exists: "expect_same"
	}
	idx: {
		"@type":    "VolumetricIndex"
		bbox:      #BBOX
		resolution: #RESOLUTION
	}
}