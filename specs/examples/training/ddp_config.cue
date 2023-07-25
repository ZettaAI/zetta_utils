"@type": "lightning_train_remote"
image: "us.gcr.io/zetta-research/zetta_utils:remote_ddp_x3"
resources: {
	memory:           "10560Mi"
	"nvidia.com/gpu": "4"
}
spec_path: "specs/examples/training/ddp.cue"
