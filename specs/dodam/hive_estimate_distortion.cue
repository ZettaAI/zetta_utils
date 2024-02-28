"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-montaging-internal-49"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_replicas:        100
local_test:             false
debug:                  false

#FOLDER: "gs://hive-tomography/pilot11-montage/exp30"

#FIELDS_PATH00: "\(#FOLDER)/fields_relaxed_00"
#FIELDS_PATH01: "\(#FOLDER)/fields_relaxed_01"
#FIELDS_PATH10: "\(#FOLDER)/fields_relaxed_10"
#FIELDS_PATH11: "\(#FOLDER)/fields_relaxed_11"

#OUTPUT_PATH: "\(#FOLDER)/lens_distortion_estimate"

target: {
	"@type":            "estimate_lens_distortion_from_csv"
	csv_path:           "./rough_montage.csv"
	info_template_path: #FIELDS_PATH00
	pad_in_res:         64
	field_paths: [#FIELDS_PATH00, #FIELDS_PATH01, #FIELDS_PATH10, #FIELDS_PATH11]
	output_path: #OUTPUT_PATH
	resolution: [4, 4, 1]
	num_tasks: 200
}
num_procs: 1
semaphores_spec: {
	"read":  1
	"cpu":   1
	"cuda":  1
	"write": 1
}
do_dryrun_estimation: false
