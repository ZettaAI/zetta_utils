// R 2UN ACED BLOCK
// INPUTS
#BASE_FOLDER: "gs://zetta_lee_fly_cns_001_alignment_temp/aced"

#IMG_PATH:     "\(#BASE_FOLDER)/coarse_x1/raw_img"
#DEFECTS_PATH: "\(#BASE_FOLDER)/coarse_x1/defect_mask"

//#ENC_PATH:      "gs://sergiy_exp/aced/demo_x0/rigid_to_elastic/raw_img_masked"
#ENC_PATH: "\(#BASE_FOLDER)/coarse_x1/encodings_masked"
#TMP_PATH: "gs://tmp_2w/temporary_layers"

//OUTPUTS
#PAIRWISE_SUFFIX: "try_x0"

#FOLDER:          "\(#BASE_FOLDER)/med_x1/\(#PAIRWISE_SUFFIX)"
#FIELDS_PATH:     "\(#FOLDER)/fields_fwd"
#FIELDS_INV_PATH: "\(#FOLDER)/fields_inv"

#IMGS_WARPED_PATH:      "\(#FOLDER)/imgs_warped"
#WARPED_BASE_ENCS_PATH: "\(#FOLDER)/base_encs_warped"
#MISALIGNMENTS_PATH:    "\(#FOLDER)/misalignments"

#TISSUE_MASK_PATH: "\(#BASE_FOLDER)/tissue_mask"

//#BASE_INFO_CHUNK: [128, 128, 1]
#BASE_INFO_CHUNK: [512, 512, 1]
#RELAX_OUTCOME_CHUNK: [32, 32, 1]

#BLOCKS: [
	//{_z_start: 0, _z_end:    452},
	//{_z_start: 451, _z_end:  901},
	//{_z_start: 900, _z_end:  1350},
	{_z_start: 6800, _z_end: 6803},
	{_z_start: 6802, _z_end: 6805},
]

#BBOX_TMPL: {
	"@type":  "BBox3D.from_coords"
	_z_start: int
	_z_end:   int
	start_coord: [0, 0, _z_start]
	end_coord: [1024, 1024, _z_end]
	resolution: [512, 512, 45]
}

#DOWNSAMPLE_FLOW: {
	"@type":     "build_subchunkable_apply_flow"
	bbox:        _
	expand_bbox: true
	processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1], [1024 * 2, 1024 * 2, 1]]
	processing_crop_pads: [[0, 0, 0], [0, 0, 0]]
	level_intermediaries_dirs: [#TMP_PATH, "~/.zutils/tmp"]

	dst_resolution: [128, 128, 45]
	op: {
		"@type": "InterpolateOperation"
		mode:    "img"
		res_change_mult: [2, 2, 1]
	}
	src: {
		"@type": "build_ts_layer"
		path:    "\(#MISALIGNMENTS_PATH)/-1"
	}
	dst: {
		"@type": "build_cv_layer"
		path:    src.path
	}
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x152"
	worker_resources: {
		memory: "18560Mi"
		//"nvidia.com/gpu": "1"
	}
	worker_replicas:        30
	batch_gap_sleep_sec:    1
	do_dryrun_estimation:   true
	local_test:             true
	worker_cluster_name:    "zutils-cns"
	worker_cluster_region:  "us-east1"
	worker_cluster_project: "zetta-lee-fly-vnc-001"

	target: {
		"@type": "mazepa.concurrent_flow"
		stages: [
			for block in #BLOCKS {
				let bbox = #BBOX_TMPL & {_z_start: block._z_start, _z_end: block._z_end}
				"@type": "mazepa.seq_flow"
				stages: [
					#DOWNSAMPLE_FLOW & {
						"bbox": bbox
					},
				]
			},
		]
	}
}

[
	#RUN_INFERENCE,
]
