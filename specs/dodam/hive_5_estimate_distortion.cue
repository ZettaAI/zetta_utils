"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-test-montaging-refactor-18"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_replicas:        50
local_test:             false
debug:                  false

#FOLDER: "gs://hive-tomography/pilot11-montage/refactor-test-0"

#FIELDS_PATH00: "\(#FOLDER)/fields_relaxed_00"
#FIELDS_PATH01: "\(#FOLDER)/fields_relaxed_01"
#FIELDS_PATH10: "\(#FOLDER)/fields_relaxed_10"
#FIELDS_PATH11: "\(#FOLDER)/fields_relaxed_11"

#OUTPUT_PATH:            "\(#FOLDER)/lens_distortion_estimate"
#TILE_REGISTRY_OUT_PATH: "dodamtesthive441_bootstrap"
#PROJECT:                "zetta-research"

target: {
	"@type": "estimate_lens_distortion_from_registry"
	tile_registry: {
		"@type":   "build_datastore_layer"
		namespace: #TILE_REGISTRY_OUT_PATH
		project:   #PROJECT
	}
	pad_in_model_res: 64
	field_paths: [#FIELDS_PATH00, #FIELDS_PATH01, #FIELDS_PATH10, #FIELDS_PATH11]
	output_path:    #OUTPUT_PATH
	tile_size_full: 5496
	model_res: [16, 16, 1]
	full_res: [4, 4, 1]
	z_start:   0
	z_stop:    2
	num_tasks: 100
}
num_procs: 1
semaphores_spec: {
	"read":  1
	"cpu":   1
	"cuda":  1
	"write": 1
}
do_dryrun_estimation: false
