"@type": "lightning_train_remote"
worker_cluster_name: "zutils-ddp-x1"
worker_cluster_region: "us-east1-d"
worker_cluster_project: "zetta-research"
worker_image: "us.gcr.io/zetta-research/zetta_utils:remote_ddp_x52"
worker_resources: {
	memory:           "8192Mi"
	"nvidia.com/gpu": "1"
}
num_nodes: 32
spec_path: "specs/examples/training/ddp.cue"
follow_logs: true

env_vars: {
    "LOGLEVEL": "INFO"
    "NCCL_SOCKET_IFNAME": "eth0"
}
