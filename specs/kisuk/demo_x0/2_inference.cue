#SRC_PATH: "gs://zetta_lee_mouse_spinal_cord_001_image/dorsal_sections/dorsal_sections_500"
#DST_PATH: "gs://tmp_2w/aff_tests/inference_x0"

#MODEL_PATH: "gs://sergiy_exp/training_artifacts/aff_tests/k3_lr0.0001_x6/last.ckpt.model.spec.json"

"@type":    "mazepa.execute_on_gcp_with_sqs"
local_test: true

target: {
	"@type": "build_subchunkable_apply_flow"
	dst_resolution: [8, 8, 45]

	start_coord: [1024 * 200, 1024 * 80, 91]
	end_coord: [1024 * 202, 1024 * 82, 111]
	coord_resolution: [4, 4, 45]

	fn: {
		"@type":      "SimpleInferenceRunner"
		model_path:   #MODEL_PATH
		unsqueeze_to: 5 // Z C X Y Z
	}
	processing_chunk_sizes: [[256, 256, 20]]
	processing_crop_pads: [[32, 32, 8]]

	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
		read_procs: [
			{"@type": "divide", "@mode": "partial", value: 127.5},
			{"@type": "add", "@mode":    "partial", value: -1},
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			num_channels: 3
			data_type:    "uint8"
		}
		info_chunk_size: [256, 256, 20]
		on_info_exists: "overwrite"
		write_procs: [
			{"@type": "squeeze", "@mode":  "partial", dim:   0},
			{"@type": "multiply", "@mode": "partial", value: 255},
			{"@type": "to_uint8", "@mode": "partial"},
		]
	}
}

// Parameters for remote execution
worker_image:    "TODO"
worker_replicas: 10
worker_resources: {
	"nvidia.com/gpu": "1"
}
