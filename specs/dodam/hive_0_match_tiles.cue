"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_cluster_region:  "us-east1"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-test-montaging-refactor-7"
worker_cluster_project: "zetta-research"
worker_cluster_name:    "zutils-x3"
worker_replicas:        100
local_test:             false
debug:                  false

#TILE_REGISTRY_PATH: "dodamtesthive441"
#PAIR_REGISTRY_PATH: "dodamtesthive441_pair_bootstrap"
#PROJECT:            "zetta-research"

target: {
	"@type": "match_tiles"
	tile_registry: {
		"@type":   "build_datastore_layer"
		namespace: #TILE_REGISTRY_PATH
		project:   #PROJECT
	}
	pair_registry: {
		"@type":   "build_datastore_layer"
		namespace: #PAIR_REGISTRY_PATH
		project:   #PROJECT
	}
	crop:                    0
	ds_factor:               12
	max_disp:                52
	z_start:                 0
	z_stop:                  2
	max_candidate_mse_ratio: 1.10
	max_num_candidates:      160
	mask_encoder_path:       "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"

	// lens_correction_model:
	// {
	//  "@type":          "LensCorrectionModel"
	//  path:             "gs://hive-tomography/pilot11-montage/exp30/lens_distortion_estimate"
	//  model_res:        4
	//  full_res:         1
	//  pad_in_model_res: 64
	//  tile_size_full:   5496
	// }
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
