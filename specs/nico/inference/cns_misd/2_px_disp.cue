#TGT_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/encodings_masked"
#SRC_PATH: "gs://zetta-research-nico/misd/cns/pairwise_enc_6150-6170/fine/-\(#Z_STEP)"

#DST_PATH: "gs://zetta-research-nico/misd/cns/inference_6150-6170_train0405/enc_misd_z\(#Z_STEP)/med_0.0px_max_0px_thr\(#THR)"

#MODEL_PATH: "gs://zetta-research-nico/training_artifacts/aced_misd_cns/thr\(#THR)_lr0.00001_z1z2_400-500_2910-2920_more_aligned_unet5_32_finetune_2/last.ckpt.model.spec.json"

#CHUNK_SIZE: [2048, 2048, 1]
#DST_INFO_CHUNK_SIZE: [2048, 2048, 1]
#RESOLUTION: [32, 32, 45]
#CROP: [128, 128, 0]
#Z_STEP: 2
#Z_START: 6150
#Z_END: 6170
#THR: 5.0

#BCUBE: {
	"@type": "BBox3D.from_coords"
	start_coord: [0 * 2048, 0 * 2048, #Z_START]
	end_coord:   [16 * 2048, 16 * 2048, #Z_END]
	resolution:  [32, 32, 45]
}

"@type":          "mazepa.execute_on_gcp_with_sqs"
worker_image:     "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230405"
worker_replicas:  10

worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "MisalignmentDetector"
			model_path: #MODEL_PATH
		}
		crop_pad: #CROP
	}
	dst_resolution: #RESOLUTION
	processing_chunk_sizes: [#CHUNK_SIZE]
	processing_crop_pads: [[0, 0, 0]]
	bbox: #BCUBE
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #TGT_PATH
		index_procs: [
			{
				"@type": "VolumetricIndexTranslator"
				offset: [0, 0, -#Z_STEP]
				resolution: [32, 32, 45]
			}
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_chunk_size:     #DST_INFO_CHUNK_SIZE
    	info_field_overrides: {
			data_type: "uint8"
		}
		on_info_exists:      "overwrite"
	}
}
