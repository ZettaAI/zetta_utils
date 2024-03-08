"@type":               "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region: "us-east1"
worker_image:          "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-montaging-internal-65"
worker_resources: {
	memory: "18560Mi" // sized for n1-highmem-4
}
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_replicas:        500
local_test:             false
debug:                  false
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		{
			"@type":            "write_files_from_csv"
			csv_path:           "./rough_montage.csv"
			info_template_path: #INFO_TEMPLATE_PATH
			base_path:          #BASE_FOLDER
			crop:               0
			resolution: [2, 2, 1]
			lens_correction_model:
			{
				"@type":        "LensCorrectionModel"
				path:           "gs://hive-tomography/pilot11-montage/exp30/lens_distortion_estimate"
				model_res:      4
				full_res:       1
				pad_in_res:     64
				tile_size_full: 5496
			}
		},
		{
			"@type": "mazepa.concurrent_flow"
			stages:
			[
				for offset in ["0_0", "0_1", "1_0", "1_1"] {
					"@type": "mazepa.sequential_flow"
					stages: [
						for res in [4, 8, 16, 32, 64, 128, 256, 512] {
							#FLOW_TMPL & {
								processing_chunk_sizes: [[1024 * 6, 1024 * 4, 4], [1024 * 2, 1024 * 2, 1]]

								op: mode: "img"
								op_kwargs: src: path: "\(#BASE_FOLDER)/\(offset)"
								dst: path: "\(#BASE_FOLDER)/\(offset)"
								dst_resolution: [res, res, 1]
							}
						},
					]
				},
			]

		},
	]
}
num_procs: 1
semaphores_spec: {
	"read":  1
	"cpu":   1
	"cuda":  1
	"write": 1
}
do_dryrun_estimation: false
#BASE_FOLDER:         "gs://hive-tomography/pilot11-tiles/rough_montaged_nocrop_tilts_exp21_0"

#INFO_TEMPLATE_PATH: "gs://hive-tomography/pilot11-tiles"

//#Z: -5

#BBOX: {
	"@type": "BBox3D.from_coords"
	//start_coord: [0, 0, #Z]
	//end_coord: [786432, 262144, start_coord[2] + 1]
	start_coord: [0, 0, -5]
	end_coord: [786432, 262144, 6]
	resolution: [1, 1, 1]

}

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	//expand_bbox_processing: true
	shrink_processing_chunk: false
	skip_intermediaries:     true
	processing_chunk_sizes:  _
	dst_resolution:          _
	op: {
		"@type":         "InterpolateOperation"
		mode:            _
		res_change_mult: _ | *[2, 2, 1]
		//res_change_mult: _ | *[8, 8, 1]
	}
	bbox: #BBOX
	op_kwargs: {
		src: {
			"@type":             "build_cv_layer"
			path:                _
			read_procs:          _ | *[]
			info_reference_path: #INFO_TEMPLATE_PATH
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _
	}

}
