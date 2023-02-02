#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/experiments/encoding_coarsener/gamma_low0.25_high4.0_prob1.0_tile_0.1_0.4_lr0.0001_post1.7_zfish_cns_64nm_unet"

#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_coarsener/tmp_ft_patch1024_post1.55_postmult1.05_lr0.0001_deep_k3_clip0.00000_equi0.5_fmt1.1_f1f2_apply_1_m3_int8_more_data_4/last.ckpt.model.spec.json"

#CHUNK_SIZE: [1024, 1024, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#RESOLUTION: [64, 64, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [10240, 0, 2700]
	end_coord: [22528, 28672, 3200]
	resolution: [32, 32, 45]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230131_unet"
worker_replicas:  100
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

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