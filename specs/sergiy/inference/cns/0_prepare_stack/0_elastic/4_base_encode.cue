// #SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
//#SRC_PATH: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img"

//#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/elastic_m3_m9_v1"
#SRC_PATH: "gs://sergiy_exp/aced/cns/giber_x0_enc/imgs_warped/-2"
#DST_PATH: "gs://sergiy_exp/aced/cns/giber_x0_enc/imgs_warped/-2_enc"

#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_encodings/gamma_low0.75_high1.5_prob1.0_tile_0.0_0.2_lr0.00002_post1.8_cns_all/last.ckpt.model.spec.json"

#CHUNK_SIZE: [2048, 2048, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#RESOLUTION: [32, 32, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2500]
	end_coord: [32768, 32768, 2525]
	resolution: [32, 32, 45]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x85"
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
		crop_pad: [32, 32, 0]
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
		"@type":    "VolumetricIndex"
		bbox:       #BBOX
		resolution: #RESOLUTION
	}
}
