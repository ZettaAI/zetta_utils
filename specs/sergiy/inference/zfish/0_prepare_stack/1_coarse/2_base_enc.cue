#SRC_PATH: "gs://zfish_unaligned/coarse_x0/raw_img_masked"
#DST_PATH: "gs://zfish_unaligned/coarse_x0/base_enc_x0"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"

#CHUNK_SIZE: [2048, 2048, 1]

#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2950]
	end_coord: [2048, 2048, 3100]
	resolution: [512, 512, 30]
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x89"
worker_replicas: 40
worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "build_subchunkable_apply_flow"
	processing_chunk_sizes: [[2048, 2048, 1]]
	processing_crop_pads: [[128, 128, 0]]
	temp_layers_dirs: ["file://~.zutils/tmp_layers"]
	dst_resolution: [32, 32, 30]
	bbox: #BBOX

	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":     "BaseEncoder"
			model_path:  #MODEL_PATH
			uint_output: true
		}
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
			data_type: "uint8"
		}
		on_info_exists: "overwrite"
	}
}
