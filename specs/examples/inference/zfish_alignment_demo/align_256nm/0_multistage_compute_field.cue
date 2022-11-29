import "path"

#VERSION:       "x1"
#RIGIDITY:      50
#XY_RESOLUTION: 256
#SRC_PATH:      "gs://zfish_unaligned/coarse_x0/encodings"
#DST_PATH:      "gs://sergiy_exp/aced/zfish/alignment_\(#XY_RESOLUTION)nm/field_\(#VERSION)"

"@type": "mazepa_execute"
target: {
	"@type": "build_compute_field_multistage_flow"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [0, 0, 3001]
		end_coord: [2048, 2048, 3002]
		resolution: [256, 256, 30]
	}
	stages: [
		{
			"@type": "ComputeFieldStage"
			dst_resolution: [#XY_RESOLUTION, #XY_RESOLUTION, 30]
			task_factory: {
				"@type": "VolumetricCallableTaskFactory"
				fn: {
					"@type": "align_with_online_finetunner"
					"@mode": "partial"
					sm:      #RIGIDITY
				}
				crop: [128, 128, 0]
			}
			chunk_size: [2048, 2048, 1]
		},
	]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	tgt_offset: [0, 0, -1]
	tgt: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "expect_same"
	}
	tmp_layer_dir: path.Join([#DST_PATH, "tmp"])
	tmp_layer_factory: {
		"@type":             "build_cv_layer"
		"@mode":             "partial"
		info_reference_path: #SRC_PATH
		info_field_overrides: {
			"num_channels": 2
			"data_type":    "float32"
		}
		on_info_exists: "expect_same"
	}
}
