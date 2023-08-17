#IMG_PATH:         "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip2_img"
#DEFECTS_PATH:     "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"
#TISSUE_MASK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/tissue_mask"
#TMP_PATH:         "gs://tmp_2w/temporary_layers"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [384, 512, 4020]
	resolution: [1024, 1024, 30]

}

#CREATE_TISSUE_MASK: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox_processing: true
	bbox:        #BBOX
	fn: {
		"@type": "apply_mask_fn"
		"@mode": "partial"
	}
	processing_chunk_sizes: [[8 * 1024, 1024 * 8, 1]]
	dst_resolution: [32, 32, 30]

	src: {
		"@type": "build_ts_layer"
		path:    #IMG_PATH
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    "!="
				value:   0
			},
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
	masks: [
		{
			"@type": "build_ts_layer"
			path:    #DEFECTS_PATH
		},
	]
	dst: {
		"@type":             "build_cv_layer"
		path:                #TISSUE_MASK_PATH
		info_reference_path: src.path
		info_field_overrides: {
			data_type: "uint8"
		}
	}
}

#DOWNSAMPLE_FLOW_TMPL: {
	"@type":     "build_subchunkable_apply_flow"
	expand_bbox_processing: true
	processing_chunk_sizes: [[1024 * 8, 1024 * 8, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	dst_resolution: _
	op: {
		"@type": "InterpolateOperation"
		mode:    _
		res_change_mult: [2, 2, 1]
		mask_value_thr: _ | *null
	}
	bbox: _
	src: {
		"@type":    "build_ts_layer"
		path:       _
		read_procs: _ | *[]
	}
	dst: {
		"@type": "build_cv_layer"
		path:    src.path
	}
}

#DOWNSAMPLE_FLOW: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		{
			"@type": "mazepa.sequential_flow"
			stages: [
				for res in [64, 128, 256, 512, 1024] {
					#DOWNSAMPLE_FLOW_TMPL & {
						bbox: #BBOX
						op: mode:           "mask"
						op: mask_value_thr: 0.9
						src: path:          #TISSUE_MASK_PATH
						dst_resolution: [res, res, 30]
					}
				},
			]
		},
	]
}

#RUN_INFERENCE: {
	"@type": "mazepa.execute_on_gcp_with_sqs"
	//worker_image: "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:sergiy_all_p39_x140"
	worker_image:         "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x187"
	do_dryrun_estimation: true
	worker_resources: {
		memory: "18560Mi"
		//"nvidia.com/gpu": "1"
	}
	worker_replicas:        200
	local_test:             false
	worker_cluster_name:    "zutils-zfish"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-jlichtman-zebrafish-001"

	target: {
		"@type": "mazepa.sequential_flow"
		stages: [
			#CREATE_TISSUE_MASK,
			#DOWNSAMPLE_FLOW,
		]
	}
}

#RUN_INFERENCE
