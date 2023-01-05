		//#SRC_PATH: "gs://zfish_unaligned/coarse_x0/raw_img"
#SRC_PATH: "gs://sergiy_exp/aced/zfish/alignment_256_128_64_32nm/raw_warped_to_z-1_x2_masked_shift_x0"
#DST_PATH: "gs://sergiy_exp/aced/zfish/alignment_256_128_64_32nm/raw_warped_to_z-1_x2_masked_shift_x0_base_enc"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"

#CHUNK_SIZE: [2048, 2048, 1]
#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]
#RESOLUTION: [32, 32, 30]
#CROP: [128, 128, 0]
#BCUBE: {
	"@type": "BoundingCube"
	start_coord: [0, 0, 3000]
	end_coord: [2048, 2048, 3020]
	resolution: [512, 512, 30]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:inference_x17"
worker_replicas:  5
worker_lease_sec: 30
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "BaseEncoder"
			model_path: #MODEL_PATH
		}
		crop: #CROP
	}
	chunker: {
		"@type":      "VolumetricIndexChunker"
		"chunk_size": #CHUNK_SIZE
		resolution:   #RESOLUTION
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
		on_info_exists:      "overwrite"
	}
	idx: {
		"@type":    "VolumetricIndex"
		bcube:      #BCUBE
		resolution: #RESOLUTION
	}
}
