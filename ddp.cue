"@type":      "multinode_train"
execution_id: "test"
image: "us.gcr.io/zetta-research/zetta_utils:ddp_multinode_v16"
resources: {}
env_vars: {
    "LOGLEVEL": "INFO"
    "NCCL_SOCKET_IFNAME": "eth0"
    "NCCL_DEBUG": "INFO"
    "WANDB_MODE": "offline"
}

master_node_ip: "10.0.0.213"
num_nodes: 4
