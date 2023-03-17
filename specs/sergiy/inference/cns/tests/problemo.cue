// RUN ACED BLOCK

// INPUTS
#IMG_PATH:      "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"
#BASE_ENC_PATH: "TODO"
#ENC_PATH:      "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"

// MODELS
#BASE_ENC_MODEL_PATH: "gs://sergiy_exp/training_artifacts/base_encodings/ft_patch1024_post1.55_lr0.001_deep_k3_clip0.00000_equi0.5_f1f2_tileaug_x17/last.ckpt.static-1.12.1+cu102-model.jit"
#MISD_MODEL_PATH:     "gs://sergiy_exp/training_artifacts/aced_misd/zm1_zm2_thr1.0_scratch_large_custom_dset_x2/checkpoints/epoch=2-step=1524.ckpt.static-1.12.1+cu102-model.jit"

//OUTPUTS
#PAIRWISE_SUFFIX: "demo_x0"

#FOLDER:          "gs://sergiy_exp/aced/cns/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#Z_START: 2500
#Z_END:   2502

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_END]
	resolution: [512, 512, 45]
}

#INVERT_FLOW_TMPL: {
	//       "@type": "build_chunked_apply_flow"
	//       operation: {
	//        "@type": "VolumetricCallableOperation"
	//        fn: {"@type": "invert_field", "@mode": "partial"}
	//        crop_pad: [64, 64, 0]
	//       }
	//       chunker: {
	//        "@type": "VolumetricIndexChunker"
	//        chunk_size: [2048, 2048, 1]
	//       }
	//       idx: {
	//        "@type": "VolumetricIndex"
	//        bbox:    #BBOX
	//        resolution: [32, 32, 45]
	//       }
	"@type": "build_subchunkable_apply_flow"
	fn: {"@type": "invert_field", "@mode": "partial"}
	processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[64, 64, 0]]
	temp_layers_dirs: ["file://~/.zutils/cache/"]
	dst_resolution: [32, 32, 45]
	bbox: #BBOX
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		on_info_exists:      "overwrite"
	}
}

#Z_OFFSETS: [-1]
#JOINT_OFFSET_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for z_offset in #Z_OFFSETS {
			"@type": "mazepa.concurrent_flow"
			stages: [

				{
					"@type": "mazepa.seq_flow"
					stages: [
						#INVERT_FLOW_TMPL & {
							src: path: "\(#FIELDS_PATH)/\(z_offset)"
							dst: path: "\(#FIELDS_INV_PATH)/\(z_offset)"
						},
					]
				},
			]
		},
	]
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x60"
worker_resources: {
	memory:           "18560Mi"
	"nvidia.com/gpu": "1"
}
worker_replicas:      25
batch_gap_sleep_sec:  1
do_dryrun_estimation: true
local_test:           false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#JOINT_OFFSET_FLOW,
		#MATCH_OFFSETS_FLOW,

		//#RELAX_FLOW,
		//#JOINT_POST_ALIGN_FLOW,
	]
}
