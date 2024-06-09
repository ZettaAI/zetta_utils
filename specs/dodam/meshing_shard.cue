//
// Handy variables
#SEG_PATH: "gs://dodam_exp/seg_medcutout"
#MESH_DIR: "mesh_mip_1_err_40"
#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [1024 * 0, 1024 * 0, 1995]
	end_coord: [1024 * 10, 1024 * 10, 1995 + 128]
	resolution: [20, 20, 50]
}

#SEG_DB_PATH:  "dodam-med-seg-512-512-128v2"
#FRAG_DB_PATH:  "dodam-med-frag-512-512-128v2"
#PROJECT:                "zetta-research"


// Execution parameters
"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us-east1-docker.pkg.dev/zetta-research/zutils/zetta_utils:dodam-meshing-15"
worker_cluster_name:    "zutils-x3"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas: 100
num_procs: 1
semaphores_spec: {
	"read": 8
	"write": 8
	"cuda": 0
	"cpu": 8
}
local_test: true // set to `false` execute remotely
debug:      true // set to `false` execute remotely
do_dryrun_estimation: false

target: {
	"@type": "build_make_shards_flow"
	segmentation_path:    #SEG_PATH
	seg_db: {
		"@type":   "build_datastore_layer"
		namespace: #SEG_DB_PATH
		project:   #PROJECT
	}
	frag_db: {
		"@type":   "build_datastore_layer"
		namespace: #FRAG_DB_PATH
		project:   #PROJECT
	}
	num_lods: 5
	min_shards: 48
	num_shard_no_tasks: 1024
}
