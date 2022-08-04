import DSET "zetta.ai/trainig_x0/datasets:dataset_x0"

import ARCH "zetta.ai/trainig_x0/architectures:architecture_x0"

"<type>": "lightning_train"
trainer: {
	"<type>":          "LightningTrainer"
	accelerator:       "gpu"
	devices:           1
	log_every_n_steps: 1
	max_epochs:        2
	default_root_dir:  "~/tmp/training_x0"
}
train_dataloader: {
	"<type>":    "TorchDataLoader"
	batch_size:  1
	shuffle:     true
	num_workers: 0
	dataset:     DSET
}
regime: {
	"<type>": "NaiveSupervised"
	lr:       3e-3
	model:    ARCH
}
