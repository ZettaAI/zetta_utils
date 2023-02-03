// RUN ACED BLOCK

// INPUTS

#Z_START: 160
#Z_END:   170

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_END]
	resolution: [512, 512, 30]
}

#MISD_MODEL_PATH: "gs://sergiy_exp/training_artifacts/aced_misd/v2_x1/last.ckpt.model.spec.json"
#BASE_ENC_PATH:   "gs://zfish_unaligned/precoarse_x0/enc_gen3_x0"
#TGT_PATH:        "gs://zfish_unaligned/precoarse_x0/enc_gen3_x0/aligned_z-1"
#DST_PATH:        "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/misd_debug_zm1_x0"

#MISD_FLOW: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":    "MisalignmentDetector"
			model_path: #MISD_MODEL_PATH
		}
		crop_pad: [32, 32, 0]
	}
	chunker: {
		"@type": "VolumetricIndexChunker"
		chunk_size: [2048, 2048, 1]
	}
	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BBOX
		resolution: [32, 32, 30]
	}
	src: {
		"@type": "build_cv_layer"
		path:    #BASE_ENC_PATH
	}
	tgt: {
		"@type": "build_cv_layer"
		path:    #TGT_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #BASE_ENC_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		info_field_overrides: {
			"data_type": "uint8"
		}
	}
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x42"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      10
	batch_gap_sleep_sec:  1
	do_dryrun_estimation: true
	local_test:           false
	debug:                true

	target: {
		"@type": "mazepa.seq_flow"
		stages: [
			//#JOINT_OFFSET_FLOW,
			#MISD_FLOW,
			//#RELAX_FLOW,
			//#JOINT_POST_ALIGN_FLOW,
		]
	}
}

[
	#RUN_INFERENCE,
]
