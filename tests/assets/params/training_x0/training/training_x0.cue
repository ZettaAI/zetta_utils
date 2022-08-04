import DSET "zetta.ai/trainig_x0/datasets:dataset_x0"

import ARCH "zetta.ai/trainig_x0/architectures:architecture_x0"

#RUN_NAME:       "training_x0"
#TRAINING_DIR:   "~/tmp/zetta_training/\(#RUN_NAME)"
#RESUME_VERSION: 4

"<type>": "lightning_train"
trainer: {
	"<type>":         "ZettaDefaultTrainer"
	accelerator:      "gpu"
	devices:          1
	max_epochs:       2
	default_root_dir: #TRAINING_DIR
	checkpointing_kwargs: {
		every_n_steps: 100
	}
}
train_dataloader: {
	"<type>":    "TorchDataLoader"
	batch_size:  4
	shuffle:     true
	num_workers: 8
	dataset:     DSET
}
//val_dataloader:
regime: {
	"<type>": "NaiveSupervised"
	lr:       3e-2
	model:    ARCH
}
//ckpt_path: "\(#TRAINING_DIR)/lightning_logs/version_\(#RESUME_VERSION)/checkpoints/last.ckpt"
