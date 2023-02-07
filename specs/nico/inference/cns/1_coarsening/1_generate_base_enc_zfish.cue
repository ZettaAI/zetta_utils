#SRC_PATH: "gs://sergiy_exp/pairs_dsets/zfish_x0/src"
#DST_PATH: "gs://zetta-research-nico/pairs_dsets/zfish_x0/src/enc/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns/enc_rendered"

#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns/last.ckpt.model.spec.json"

#CHUNK_SIZE: [2048, 2048, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [256, 256, 0]
	end_coord: [1024, 1024, 41]
	resolution: [256, 256, 30]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_3"
worker_replicas: 10
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
		resolution: [32, 32, 30]
	}
}
