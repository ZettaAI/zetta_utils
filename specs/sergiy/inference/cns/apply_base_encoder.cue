#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v3/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
#DST_PATH: "gs://tmp_2w/cns/base_enc_x0"

#MODEL_SPEC_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x16/last.ckpt.model.spec.json"

#CHUNK_SIZE: [2048, 2048, 1]
#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]
#RESOLUTION: [32, 32, 45]
#CROP: [128, 128, 0]
#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [1024, 0, 3000]
	end_coord: [2048, 1024, 3001]
	resolution: [512, 512, 45]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:inference_x16"
worker_replicas:  5
worker_lease_sec: 30
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: true

target: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "apply_base_encoder"
			"@mode": "partial"
		}
		crop: #CROP
	}
	chunker: {
		"@type":      "VolumetricIndexChunker"
		"chunk_size": #CHUNK_SIZE
		resolution:   #RESOLUTION
	}
	model_spec_path: #MODEL_SPEC_PATH
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_chunk_size:     #DST_INFO_CHUNK_SIZE
		on_info_exists:      "overwrite"
	}
	idx: {
		"@type":    "VolumetricIndex"
		bcube:      #BCUBE
		resolution: #RESOLUTION
	}
}
