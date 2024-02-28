"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-test-montaging-refactor-8"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_replicas:        5
local_test:             false
debug:                  false

#TILE_REGISTRY_IN_PATH:  "dodamtesthive441"
#TILE_REGISTRY_OUT_PATH: "dodamtesthive441_bootstrap"
#PAIR_REGISTRY_PATH:     "dodamtesthive441_pair_bootstrap"
#PROJECT:                "zetta-research"

target: {
	"@type": "elastic_tile_placement"
	tile_registry_in: {
		"@type":   "build_datastore_layer"
		namespace: #TILE_REGISTRY_IN_PATH
		project:   #PROJECT
	}
	tile_registry_out: {
		"@type":   "build_datastore_layer"
		namespace: #TILE_REGISTRY_OUT_PATH
		project:   #PROJECT
	}
	pair_registry: {
		"@type":   "build_datastore_layer"
		namespace: #PAIR_REGISTRY_PATH
		project:   #PROJECT
	}
	std_filter:           25.0
	z_start:              0
	z_stop:               2
	min_x:                8192
	min_y:                8192
	mse_consensus_filter: 3.0
}
num_procs: 1
semaphores_spec: {
	"read":  1
	"cpu":   1
	"cuda":  1
	"write": 1
}
worker_resources: {
	memory: "18560Mi" // sized for n1-highmem-4

	"nvidia.com/gpu": "1"
}
do_dryrun_estimation: false
