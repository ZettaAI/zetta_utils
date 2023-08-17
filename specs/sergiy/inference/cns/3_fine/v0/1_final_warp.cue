#IMG_PATH:          "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"
#RIGID_IMG_PATH:    "gs://zetta_lee_fly_cns_001_alignment_temp/rigid"
#COARSE_FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"
#JOINT_FIELD_PATH:  "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)_from_rigid"

//OUTPUTS
#PAIRWISE_SUFFIX: "giber_x0_enc"

#FOLDER:          "gs://sergiy_exp/aced/cns/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/manual_misalignments"

#MATCH_OFFSETS_PATH: "\(#FOLDER)/match_offsets_manual"

#AFIELD_PATH:             "\(#FOLDER)/afield\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_PATH:        "\(#FOLDER)/img_aligned\(#RELAXATION_SUFFIX)"
#IMG_MASK_PATH:           "\(#FOLDER)/img_mask\(#RELAXATION_SUFFIX)"
#IMG_ALIGNED_MASKED_PATH: "\(#FOLDER)/img_aligned_masked\(#RELAXATION_SUFFIX)"

#CF_INFO_CHUNK: [512, 512, 1]
#AFIELD_INFO_CHUNK: [512, 512, 1]
#RELAXATION_CHUNK: [512, 512, #Z_END - #Z_START]
#RELAXATION_FIX:  "first"
#RELAXATION_ITER: 500
#RELAXATION_LR:   0.3
#RELAXATION_RIG:  200

#Z_START: 3300
#Z_END:   3400

//#RELAXATION_SUFFIX: "_fix\(#RELAXATION_FIX)_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_z\(#Z_START)-\(#Z_END)"
#RELAXATION_SUFFIX: "_try_x4_iter\(#RELAXATION_ITER)_rig\(#RELAXATION_RIG)_lr\(#RELAXATION_LR)"

#BBOX: {
	"@type": "BBox3D.from_coords"
	//start_coord: [512 + 256, 512, #Zc:_START]
	//end_coord: [1024, 512 + 256, #Z_END]
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_END]
	resolution: [512, 512, 45]
}

#TITLE: #PAIRWISE_SUFFIX
#MAKE_NG_URL: {
	"@type": "make_ng_link"
	title:   #TITLE
	position: [50000, 60000, #Z_START + 1]
	scale_bar_nm: 30000
	layers: [
		["precoarse_img", "image", "precomputed://\(#IMG_PATH)"],
		["+1 \(#PAIRWISE_SUFFIX)", "image", "precomputed://\(#IMGS_WARPED_PATH)/+1"],
		["aligned \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_PATH)"],
		["aligned masked \(#RELAXATION_SUFFIX)", "image", "precomputed://\(#IMG_ALIGNED_MASKED_PATH)"],
	]
}

#NOT_FIRST_SECTION_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START + 1]
	end_coord: [ 2048, 2048, #Z_END]
	resolution: [512, 512, 45]
}
#FIRST_SECTION_BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, #Z_START]
	end_coord: [2048, 2048, #Z_START + 1]
	resolution: [512, 512, 45]
}

#WARP_FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    _
	}
	processing_crop_pads: [[256, 256, 0]]
	processing_chunk_sizes: [[2048, 2048, 1]]
	//chunk_size: [512, 512, 1]
	bbox:           #BBOX
	dst_resolution: _
	src: {
		"@type":               "build_cv_layer"
		path:                  _
		data_resolution?:      _
		interpolation_mode?:   _
		allow_slice_rounding?: _
		read_procs?:           _
		index_procs?:          _ | *[]
	}
	field: {
		"@type":             "build_cv_layer"
		path:                _
		data_resolution?:    _
		interpolation_mode?: _
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: _
		on_info_exists:      "overwrite"
		write_procs?:        _
		index_procs?:        _ | *[]
	}
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x107"
	worker_resources: {
		memory: "18560Mi"
		//"nvidia.com/gpu": "1"
	}
	worker_replicas:      100
	batch_gap_sleep_sec:  0.4
	do_dryrun_estimation: true
	local_test:           false

	target: {
		"@type": "mazepa.sequential_flow"
		stages: [
			#WARP_FLOW_TMPL & {
				op: mode: "field"
				dst_resolution: [32, 32, 45]
				bbox: #BBOX
				field: path: #AFIELD_PATH
				src: {
					data_resolution: [256, 256, 45]
					path:                 #COARSE_FIELD_PATH
					interpolation_mode:   "field"
					allow_slice_rounding: true
				}
				dst: path:                #JOINT_FIELD_PATH
				dst: info_reference_path: #AFIELD_PATH
			},

			#WARP_FLOW_TMPL & {
				op: mode: "img"
				dst_resolution: [16, 16, 45]
				src: path: #RIGID_IMG_PATH
				field: {
					path: #JOINT_FIELD_PATH
					data_resolution: [32, 32, 45]
					interpolation_mode: "field"
				}
				dst: path:                #IMG_ALIGNED_PATH
				dst: info_reference_path: #IMG_PATH
			},
		]
	}
}

[
	//#MAKE_NG_URL,
	#RUN_INFERENCE,
	//#MAKE_NG_URL,
]
