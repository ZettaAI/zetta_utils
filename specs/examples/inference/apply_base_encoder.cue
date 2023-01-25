#SRC_PATH: "gs://zfish_unaligned/coarse_x0/raw_img"
#DST_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x0"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"

#CHUNK_SIZE: [2048, 2048, 1]
#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]
#RESOLUTION: [32, 32, 30]
#CROP: [128, 128, 0]
#BCUBE: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3000]
	end_coord: [2048, 2048, 3020]
	resolution: [512, 512, 30]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_inference_x60_p39"
worker_replicas: 1

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
		crop_pad: #CROP
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
		bbox:       #BCUBE
		resolution: #RESOLUTION
	}
}
