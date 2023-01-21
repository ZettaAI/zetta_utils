#VERSION:              "x1"
#RIGIDITY:             50
#NUM_ITER:             200
#STAGE0_XY_RESOLUTION: 256
#STAGE1_XY_RESOLUTION: 64
#SRC_PATH:             "gs://zfish_unaligned/coarse_x0/encodings"
#DST_PATH:             "gs://sergiy_exp/aced/zfish/alignment_\(#STAGE0_XY_RESOLUTION)_\(#STAGE1_XY_RESOLUTION)nm/field_\(#VERSION)"

"@type": "mazepa.execute"
target: {
	"@type": "build_compute_field_multistage_flow"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [0, 0, 3002]
		end_coord: [2048, 2048, 3019]
		resolution: [256, 256, 30]
	}
	stages: [
		{
			"@type": "ComputeFieldStage"
			dst_resolution: [#STAGE0_XY_RESOLUTION, #STAGE0_XY_RESOLUTION, 30]
			operation: {
				"@type": "VolumetricCallableOperation"
				fn: {
					"@type":  "align_with_online_finetuner"
					"@mode":  "partial"
					sm:       #RIGIDITY
					num_iter: #NUM_ITER
				}
				crop: [128, 128, 0]
			}
			chunk_size: [2048, 2048, 1]
		},
		{
			"@type": "ComputeFieldStage"
			dst_resolution: [#STAGE1_XY_RESOLUTION, #STAGE1_XY_RESOLUTION, 30]
			operation: {
				"@type": "VolumetricCallableOperation"
				fn: {
					"@type":  "align_with_online_finetuner"
					"@mode":  "partial"
					sm:       #RIGIDITY
					num_iter: #NUM_ITER
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
	tmp_layer_dir: "\(#DST_PATH)/tmp"
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
