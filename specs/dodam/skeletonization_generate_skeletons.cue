//
// Handy variables
#SEG_PATH: "gs://dodam_exp/seg_bigcutout"

// Read the `mesh`dir` requirement in mesh_generation.py
#SKELETON_DIR: "skeletons_mip_0_test0"
#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [1024 * 5, 1024 * 5, 1995]
	end_coord: [1024 * 9, 1024 * 9, 1995 + 512]
	resolution: [20, 20, 50]
}

#SEG_DB_PATH:  "dodam-skel-med-seg-512-512-256v2"
#FRAG_DB_PATH: "dodam-skel-med-frag-512-512-256v2"
#PROJECT:      "zetta-research"

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
num_procs:       16
semaphores_spec: {
	"read":  1
	"write": 1
	"cuda":  0
	"cpu":   1
}

local_test:           true // set to `false` execute remotely
do_dryrun_estimation: false

target: {
	"@type": "build_generate_skeletons_flow"
	bbox:    #BBOX
	seg_resolution: [20, 20, 50]
	frag_chunk_size: [512, 512, 256]
	segmentation: {
		"@type": "build_cv_layer"
		path:    #SEG_PATH
	}
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
	skeleton_dir:       #SKELETON_DIR
	min_shards:         8
	num_shard_no_tasks: 16
}
