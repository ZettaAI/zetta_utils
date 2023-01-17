#SRC_PATH: "gs://zfish_unaligned/precoarse_x0/raw_masked"
#DST_PATH: "gs://zfish_unaligned/precoarse_x0/base_enc_x0"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"

#CHUNK_SIZE: [2048, 2048, 1]
#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 0]
	end_coord: [1024, 1024, 50]
	resolution: [512, 512, 30]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x22"
worker_replicas:  10
worker_lease_sec: 10
worker_resources: {
	"nvidia.com/gpu": "1"
}
batch_gap_sleep_sec: 1

local_test: false

target: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseEncoder"
			model_path: #MODEL_PATH
		}
		crop: [128, 128, 0]
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
		on_info_exists:      "expect_same"
	}
	idx: {
		"@type": "VolumetricIndex"
		bcube:   #BCUBE
		resolution: [32, 32, 30]
	}
}
