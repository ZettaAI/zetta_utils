#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"

//#DST_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x1"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/enc/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0002_x13/enc_rendered"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/tmp_gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0002_x13/last.ckpt.static-1.12.1+cu102-model.jit"

#CHUNK_SIZE: [2048, 2048, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3000]
	end_coord: [512, 576, 3003]
	resolution: [2048, 2048, 45]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:py3.10_torch_1.13.1_cu11.7_zu20230131"
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
	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BBOX
		resolution: [32, 32, 45]
	}
}
