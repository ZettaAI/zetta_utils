"@type": "lightning_train_remote"
worker_image: "us.gcr.io/zetta-research/zetta_utils:remote_ddp_x8"
worker_resources: {
	memory:           "10560Mi"
	"nvidia.com/gpu": "4"
}
spec_path: "specs/examples/training/ddp.cue"
follow_logs: true
