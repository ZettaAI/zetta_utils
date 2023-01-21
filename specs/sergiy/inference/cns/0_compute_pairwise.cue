#VERSION:  "v0"
#RIGIDITY: 50
#NUM_ITER: 150
#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/encodings/rigid_v2"
#DST_PATH: "gs://sergiy_exp/cns/alignment_tmp_\(#VERSION)/field"
#BCUBE: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2508]
	end_coord: [2048, 2048, 2509]
	resolution: [512, 512, 45]
}
#RESOLUTIONS: [
	[512, 512, 45],
	[256, 256, 45],
]
#STAGE_TMPL: {
	"@type":        "ComputeFieldStage"
	dst_resolution: _
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
}
"@type": "mazepa.execute"
target: {
	"@type": "build_compute_field_multistage_flow"
	bbox:   #BCUBE
	stages: [
		for res in #RESOLUTIONS {
			#STAGE_TMPL & {'dst_resolution': res}
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
