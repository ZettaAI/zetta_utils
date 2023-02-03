#Z_START: 157
#Z_END:   160

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [56 * 1024, 64 * 1024, #Z_START]
	end_coord: [72 * 1024, 80 * 1024, #Z_END]
	resolution: [4, 4, 30]
}

#DST_PATH:   "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/inverse_quality_debug_x"
#INV_PATH:   "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/fields_fwd/-1"
#FIELD_PATH: "gs://sergiy_exp/aced/zfish/late_jan_cutout_x0/fields_bwd/-1"

#FLOW: {
	"@type": "build_warp_flow"
	mode:    "field"
	crop_pad: [32, 32, 0]
	chunk_size: [1024, 1024, 1]
	bbox: #BBOX
	dst_resolution: [32, 32, 30]
	src: {
		"@type": "build_cv_layer"
		path:    #FIELD_PATH
	}
	field: {
		"@type": "build_cv_layer"
		path:    #INV_PATH
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #FIELD_PATH
		info_chunk_size: [1024, 1024, 1]
		on_info_exists: "overwrite"
		info_field_overrides: {
			num_channels: "1"
			data_type:    "uint8"
		}
		write_procs: [
			{
				"@type": "abs"
				"@mode": "partial"
			},
			{
				"@type":   "reduce"
				"@mode":   "partial"
				pattern:   "2 X Y Z -> 1 X Y Z"
				reduction: "max"
			},
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
}

#RUN_INFERENCE: {
	"@type":      "mazepa.execute_on_gcp_with_sqs"
	worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x42"
	worker_resources: {
		memory:           "18560Mi"
		"nvidia.com/gpu": "1"
	}
	worker_replicas:      10
	batch_gap_sleep_sec:  1
	do_dryrun_estimation: true
	local_test:           true
	debug:                true

	target: {
		"@type": "mazepa.seq_flow"
		stages: [
			#FLOW,
		]
	}
}

[
	#RUN_INFERENCE,
]
