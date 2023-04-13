#DEFECTS_PATH: "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/defect_mask"
#RESIN_PATH:   "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/resin_mask"

#IMG_PATH:        "gs://zetta-research-nico/pairs_dsets/cns_x0_3406-3410"
#IMG_MASKED_PATH: "gs://zetta-research-nico/pairs_dsets/cns_x0_3406-3410_masked"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3405]
	end_coord: [2048, 2048, 3406]
	resolution: [512, 512, 45]
}
#BIG_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 3405]
	end_coord: [1024 * 8, 1024 * 9, 3406]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type":        "build_subchunkable_apply_flow"
	op: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "apply_mask_fn"
			"@mode": "partial"
		}
	}
	processing_chunk_sizes: _
	processing_crop_pads: [[0, 0, 0]]
	dst_resolution: _
    expand_bbox: true
	src: {
		"@type": "build_cv_layer"
		path:    _
	}
	masks: [
		{
			"@type": "build_cv_layer"
			path:    #DEFECTS_PATH
			read_procs: [
				{
					"@type": "coarsen_mask"
					"@mode": "partial"
					width:   1
				},

			]
		},
		{
			"@type": "build_cv_layer"
			path:    #RESIN_PATH
			data_resolution: [256, 256, 45]
			interpolation_mode: "mask"
		},
	]

	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		on_info_exists:      "expect_same"
	}
	bbox: _
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:nico_py3.9_20230329"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     20
batch_gap_sleep_sec: 0.05

local_test: false

target: {
	"@type": "mazepa.seq_flow"
	stages: [
		for res in [32, 64, 128] {
			#FLOW_TMPL & {
				processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1]]
				bbox: #BBOX
				dst_resolution: [res, res, 45]
				src: path: #IMG_PATH
				dst: path: #IMG_MASKED_PATH
			}
		},
		for res in [256, 512] {
			#FLOW_TMPL & {
				processing_chunk_sizes: [[1024 * 2, 1024 * 2, 1]]
				bbox: #BBOX
				dst_resolution: [res, res, 45]
				src: path: #IMG_PATH
				dst: path: #IMG_MASKED_PATH
			}
		},
		for res in [1024, 2048, 4096] {
			#FLOW_TMPL & {
				processing_chunk_sizes: [[1024, 1152, 1]]
				bbox: #BIG_BBOX
				dst_resolution: [res, res, 45]
				src: path: #IMG_PATH
				dst: path: #IMG_MASKED_PATH
			}

		},

	]
}
