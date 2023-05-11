"@type":      "multinode_train"
execution_id: "test"
image: "us.gcr.io/zetta-research/zetta_utils:ddp_multinode_v27"
resources: {}
env_vars: {
    "LOGLEVEL": "INFO"
    "NCCL_SOCKET_IFNAME": "eth0"
    "WANDB_MODE": "disabled"
}

master_node_ip: "10.0.0.213"
num_nodes: 4
