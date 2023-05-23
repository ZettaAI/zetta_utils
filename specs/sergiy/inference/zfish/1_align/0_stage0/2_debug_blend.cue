// R 2UN ACED BLOCK
// INPUTS

#IMG_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip3_img_defects_masked"
#DST_PATH: "gs://tmp_2w/demo/subch_debug_x0"
#TMP_PATH: "gs://tmp_2w/tmp/"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 1000]
	end_coord: [384, 512, 1001]
	resolution: [1024, 1024, 30]
}

#TEST_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":      "lambda"
		"lambda_str": "lambda src: (src != src).float()"
	}
	expand_bbox: true
	dst_resolution: [512, 512, 30]
	bbox: #BBOX

	processing_chunk_sizes: [[32, 32, 1]]
	max_reduction_chunk_sizes: [32, 32, 1]
	processing_crop_pads: [[32, 32, 0]]
	processing_blend_pads: [[8, 8, 0]]
	level_intermediaries_dirs: [#TMP_PATH]
	src: {
		"@type": "build_cv_layer"
		path:    #IMG_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #IMG_PATH
		info_field_overrides: {
			data_type: "float32"
		}
		info_chunk_size: [32, 32, 1]
		on_info_exists: "overwrite"
	}
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x193"

	worker_resources: {
		memory: "18560Mi"
		//"nvidia.com/gpu": "1"
	}
	worker_replicas:        200
	do_dryrun_estimation:   true
	local_test:             true
	worker_cluster_name:    "zutils-zfish"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-jlichtman-zebrafish-001"

	target: #TEST_FLOW
}

#RUN_INFERENCE
