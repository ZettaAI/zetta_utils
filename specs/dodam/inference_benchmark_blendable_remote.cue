#IMAGE_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment/fine_full_v1/img"
#IMAGE_RESOLUTION: [16, 16, 30]

#IMAGE_MASK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment/fine_full_v1/img/input_mask"
#IMAGE_MASK_RESOLUTION: [64, 64, 30]
#INVERT_IMAGE_MASK: true

#OUTPUT_MASK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment/fine_full_v1/img/output_mask"
#OUTPUT_MASK_RESOLUTION: [64, 64, 30]
#INVERT_OUTPUT_MASK: true

#MODEL_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/models/zfish_aff_mye+ecs.onnx"
#INPUT_PAD: [16, 16, 2]
#MYELIN_MASK_THRESHOLD: 0.2

#DST_PATH:   "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/benchoutput_20/"
#TEMP_PATH1: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/dodam_exp/benchoutput_20/temp/"
#TEMP_PATH0: "file:///tmp/zetta_cvols/"

"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:dodam_subchunkable_x15"
worker_replicas: 4

worker_resources: {
	"nvidia.com/gpu": "1"
}

local_test: false

target: {
	"@type": "build_blendable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type":               "run_affinities_inference"
			"@mode":               "partial"
			model_path:            #MODEL_PATH
			myelin_mask_threshold: #MYELIN_MASK_THRESHOLD
		}

	}
	start_coord: [9544, 15496, 0]
	end_coord: [9832, 15784, 32]
	coord_resolution: [16, 16, 30]

	dst_resolution: #IMAGE_RESOLUTION

	// these are the args that need to be duplicated for all the levels
	// expand singletons, raise exception if lengths not same
	processing_chunk_size: [48, 48, 8]

	processing_crop_pad: [16, 16, 2]
	processing_blend_pad: [24, 24, 4]
	processing_blend_mode: "quadratic"

	fov_crop_pad: [0, 0, 0]

	max_reduction_chunk_size: [4096, 4096, 16]

	temp_layers_dir: #TEMP_PATH1

	image: {
		"@type": "build_cv_layer"
		path:    #IMAGE_PATH
		read_procs: [
			{
				"@mode":   "partial"
				"@type":   "rearrange"
				"pattern": "c x y z -> c z x y"
			},
		]
	}
	image_mask: {
		"@type":            "build_cv_layer"
		path:               #IMAGE_MASK_PATH
		data_resolution:    #IMAGE_MASK_RESOLUTION
		interpolation_mode: "mask"
		read_procs: [
			{
				"@type": "InvertProcessor"
				invert:  #INVERT_IMAGE_MASK
			},
			{
				"@mode":   "partial"
				"@type":   "rearrange"
				"pattern": "c x y z -> c z x y"
			},
		]
	}
	output_mask: {
		"@type":            "build_cv_layer"
		path:               #OUTPUT_MASK_PATH
		data_resolution:    #OUTPUT_MASK_RESOLUTION
		interpolation_mode: "mask"
		read_procs: [
			{
				"@type": "InvertProcessor"
				invert:  #INVERT_OUTPUT_MASK
			},
			{
				"@mode":   "partial"
				"@type":   "rearrange"
				"pattern": "c x y z -> c z x y"
			},
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #IMAGE_PATH
		info_field_overrides: {
			"num_channels": 3
			"data_type":    "float32"
		}
		on_info_exists: "overwrite"
	}
}
