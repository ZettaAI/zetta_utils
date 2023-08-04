// R 2UN ACED BLOCK
// INPUTS
#BASE_FOLDER: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/aced"

#IMG_PATH:         "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/mip3_img_defects_masked"
#DEFECTS_PATH:     "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/defects_binarized"
#TISSUE_MASK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/affine/v3_phase2/tissue_mask"

#TMP_PATH: "gs://tmp_2w/temporary_layers"

//OUTPUTS
#PAIRWISE_SUFFIX: "final_x0"

#FOLDER:          "\(#BASE_FOLDER)/med_x1/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

//#MATCH_OFFSETS_BASE: "\(#FOLDER)/match_offsets_z\(#Z_START)_\(#Z_END)"
//#BASE_INFO_CHUNK: [128, 128, 1]
#BASE_INFO_CHUNK: [512, 512, 1]

#RELAX_OUTCOME_CHUNK: [32, 32, 1]

//#Z_END:   746

#BLOCKS: [
	{_z_start: 1142, _z_end: 1148},
	//{_z_start: 0, _z_end:    452},
	//{_z_start: 451, _z_end:  1350},
	//{_z_start: 1349, _z_end: 2251},
	//{_z_start: 2250, _z_end: 3155},
	// {_z_start: 3154, _z_end: 4051},
	// {_z_start: 4050, _z_end: 4953},
	// {_z_start: 4952, _z_end: 5853},
	// {_z_start: 5852, _z_end: 6751},
	// {_z_start: 6750, _z_end: 7051},
]

#BBOX_TMPL: {
	"@type":  "BBox3D.from_coords"
	_z_start: int
	_z_end:   int
	start_coord: [0, 0, _z_start]
	end_coord: [382, 512, _z_end]
	//start_coord: [0, 0, _z_start]
	//end_coord: [384, 512, _z_end]
	resolution: [1024, 1024, 30]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [52189, 67314, 1025]
	scale_bar_nm: 30000
	layers: [
		["precoarse_img", "image", "precomputed://\(#IMG_PATH)"],
		//["-1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/-1"],
		//["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
	]
}

#TEST_MASK_FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":      "lambda"
		"lambda_str": "lambda src: src"
	}
	expand_bbox_processing: true
	dst_resolution: [512, 512, 30]
	bbox: _

	processing_chunk_sizes: [[32, 32, 1]]
	max_reduction_chunk_sizes: [32, 32, 1]
	processing_crop_pads: [[40, 40, 0]]
	level_intermediaries_dirs: [#TMP_PATH]
	//              processing_chunk_sizes: [[32, 32, #Z_END - #Z_START], [28, 28, #Z_END - #Z_START]]
	//              max_reduction_chunk_sizes: [128, 128, #Z_END - #Z_START]
	//              processing_crop_pads: [[0, 0, 0], [16, 16, 0]]
	//              processing_blend_pads: [[12, 12, 0], [12, 12, 0]]
	//level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]
	src: {
		"@type": "build_ts_layer"
		path:    "\(#MISALIGNMENTS_PATH)/-1"
		read_procs: [
			{
				"@type": "compare"
				"@mode": "partial"
				mode:    ">="
				value:   70
			},
			{
				"@type": "filter_cc"
				"@mode": "partial"
				thr:     20
				mode:    "keep_large"
			},
			{
				"@type": "binary_closing"
				"@mode": "partial"
				width:   12
			},
			{
				"@type": "coarsen"
				"@mode": "partial"
				width:   2
			},
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "\(#MISALIGNMENTS_PATH)/-1_debug_filtered"
		info_reference_path: "\(#MISALIGNMENTS_PATH)/-1"
		info_chunk_size:     #RELAX_OUTCOME_CHUNK
		on_info_exists:      "overwrite"
	}
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x194"

	worker_resources: {
		memory: "18560Mi"
		//"nvidia.com/gpu": "1"
	}
	worker_replicas:        40
	do_dryrun_estimation:   true
	local_test:             false
	worker_cluster_name:    "zutils-zfish"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-jlichtman-zebrafish-001"

	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for block in #BLOCKS {
				let bbox = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
				"@type": "mazepa.seq_flow"
				stages: [
					// #JOINT_OFFSET_FLOW & {
					//  _bbox: bbox
					// },
					// #CREATE_TISSUE_MASK & {
					//  'bbox': bbox
					// },
					// #DOWNSAMPLE_FLOW & {
					//  _bbox: bbox
					// }
					//#MATCH_OFFSETS_FLOW & {'bbox': bbox},
					//#RELAX_FLOW & {'bbox':     bbox},
					//#POST_ALIGN_FLOW & {_bbox: bbox},
					#TEST_MASK_FLOW & {"bbox": bbox},
				]
			},
		]
	}
}

[
	#MAKE_NG_URL,
	#RUN_INFERENCE,
	#MAKE_NG_URL,
]
