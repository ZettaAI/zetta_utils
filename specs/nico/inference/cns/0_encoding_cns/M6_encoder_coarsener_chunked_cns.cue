// #SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/M7_500xSM200_M6_500xSM200_M5_500xSM200_M4_250xSM200_M3_250xSM200_VV3_CT2.5_BS10/mip1/img/img_rendered"
#SRC_PATH: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img"
#DST_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/elastic_m3_m9_v1"

#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/base_coarsener_cns/4_M3_M6_conv3_unet1_lr0.0001_equi0.5_post1.1_fmt0.8_cns_all/last.ckpt.model.spec.json"

#CHUNK_SIZE: [1024, 1024, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#RES_MULT: 8
#RESOLUTION: [#RES_MULT * 32, #RES_MULT * 32, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3300]
	end_coord: [32768, 32768, 3501]
	resolution: [32, 32, 45]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:py3.9_torch_1.13.1_cu11.7_zu20230213_2"
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
			"@type":    "BaseCoarsener"
			model_path: #MODEL_PATH
			tile_pad_in: 128
			tile_size: 1024
			ds_factor: #RES_MULT
		}
		crop_pad: [128, 128, 0]
		res_change_mult: [#RES_MULT, #RES_MULT, 1]
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