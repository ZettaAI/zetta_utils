"@type": "lightning_train_remote"
worker_cluster_name: "zutils-x3"
worker_cluster_region: "us-east1"
worker_cluster_project: "zetta-research"
// worker_image: "us.gcr.io/zetta-research/zetta_utils:remote_ddp_x52"
worker_image: "us.gcr.io/zetta-research/zetta_utils:kisuk_main_py3.10_20231030_x0"
// worker_image: "us.gcr.io/zetta-research/zetta_utils:kisuk_py3.10_20231030_x1"
worker_resources: {
    // memory:           "32768Mi"
	"nvidia.com/gpu": "4"
}
worker_resource_requests: {
	"memory": "8192Mi"
}
num_nodes: 1
spec_path: "specs/kisuk/ddp/ddp.cue"
follow_logs: true

env_vars: {
    "LOGLEVEL": "INFO"
    "NCCL_SOCKET_IFNAME": "eth0"
    "NCCL_DEBUG": "INFO"
    "TORCH_DISTRIBUTED_DEBUG": "DETAIL"
}
