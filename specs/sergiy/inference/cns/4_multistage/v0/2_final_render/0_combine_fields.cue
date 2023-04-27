#FOLDER:              "gs://zetta_lee_fly_cns_001_alignment_temp/aced/med_x1/try_x0"
#SUFFIX:              "_x0"
#FINE_FIELD_COMBINED: "\(#FOLDER)/afield_fine_combined\(#SUFFIX)"

#COARSE_FIELD: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x1/field"
#MED_FIELD:    "\(#FOLDER)/afield_1024nm_try_x14_iter8000_rig0.5_lr0.001_clip0.01"
#FINE_FIELD:   "\(#FOLDER)/afield_stage1_64nm_try_x1_iter300_rig200_lr0.001"

#COARSE_FIELD_RESOLUTION: [256, 256, 45]
#MED_FIELD_RESOLUTION: [1024, 1024, 45]
#FINE_FIELD_RESOLUTION: [64, 64, 45]

#MED_FIELD_COMBINED: "\(#FOLDER)/afield_med_combined\(#SUFFIX)"
#MED_FIELD_COMBINED_RESOLUTION: [256, 256, 45]

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 2701]
	end_coord: [2048, 2048, 3155]
	resolution: [512, 512, 45]
}

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	op: {
		"@type": "WarpOperation"
		mode:    "field"
	}
	shrink_processing_chunk: true
	processing_chunk_sizes: [[8 * 1024, 8 * 1024, 1], [2048, 2048, 1]]
	processing_crop_pads: [[0, 0, 0], [512, 512, 0]]
	level_intermediaries_dirs: ["~/.zutils/tmp", "~/.zutils/tmp"]
	dst_resolution: _

	bbox: #BBOX
	src: {
		"@type":            "build_cv_layer"
		path:               _
		data_resolution:    _
		interpolation_mode: "field"
	}
	field: {
		"@type":            "build_ts_layer"
		path:               _
		data_resolution:    _
		interpolation_mode: "field"
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                _
		info_reference_path: src.path
		info_chunk_size: [1024, 1024, 1]
		//on_info_exists:      "overwrite"
	}
}

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x184"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_replicas:        250
local_test:             false
target: {
	"@type": "mazepa.seq_flow"
	stages: [
		#FLOW_TMPL & {
			src: path:            #COARSE_FIELD
			src: data_resolution: #COARSE_FIELD_RESOLUTION

			field: path:            #MED_FIELD
			field: data_resolution: #MED_FIELD_RESOLUTION
			dst_resolution: #MED_FIELD_COMBINED_RESOLUTION

			dst: path: #MED_FIELD_COMBINED
		},
		#FLOW_TMPL & {
			src: path:            #MED_FIELD_COMBINED
			src: data_resolution: #MED_FIELD_COMBINED_RESOLUTION

			field: path:            #FINE_FIELD
			field: data_resolution: #FINE_FIELD_RESOLUTION
			dst_resolution: #FINE_FIELD_RESOLUTION

			dst: path: #FINE_FIELD_COMBINED
		},
	]
}
