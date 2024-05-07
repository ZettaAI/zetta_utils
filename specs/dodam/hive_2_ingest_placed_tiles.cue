"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-test-montaging-refactor-9"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_resources: {
	memory: "18560Mi" // sized for n1-highmem-4
}
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_replicas:        100
local_test:             false
debug:                  false

#TILE_REGISTRY_OUT_PATH: "dodamtesthive441_bootstrap"
#PROJECT:                "zetta-research"

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		{
			"@type":            "ingest_from_registry"
			info_template_path: #INFO_TEMPLATE_PATH
			base_path:          #BASE_FOLDER
			tile_registry: {
				"@type":   "build_datastore_layer"
				namespace: #TILE_REGISTRY_OUT_PATH
				project:   #PROJECT
			}
			crop: 0
			resolution: [4, 4, 1]
			z_start:   0
			z_stop:    2
			num_tasks: 100
			//   lens_correction_model:
			//   {
			//    "@type":        "LensCorrectionModel"
			//    path:           "gs://hive-tomography/pilot11-montage/exp30/lens_distortion_estimate"
			//    model_res:      4
			//    full_res:       1
			//    pad_in_res:     64
			//    tile_size_full: 5496
			//   }
		},
		{
			"@type": "mazepa.concurrent_flow"
			stages:
			[
				for offset in ["0_0", "0_1", "1_0", "1_1"] {
					"@type": "mazepa.sequential_flow"
					stages: [
						for res in [8, 16, 32, 64, 128, 256, 512, 1024] {
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
#BASE_FOLDER:         "gs://hive-tomography/pilot11-tiles/refactor-test-0"

#INFO_TEMPLATE_PATH: "gs://hive-tomography/pilot11-tiles"

//#Z: -5

#BBOX: {
	"@type": "BBox3D.from_coords"
	//start_coord: [0, 0, #Z]
	//end_coord: [786432, 262144, start_coord[2] + 1]
	start_coord: [0, 0, 0]
	end_coord: [131072, 131072, 2]
	resolution: [4, 4, 1]

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
