import DSET "zetta.ai/trainig_x0/datasets:dataset_x0"

import ARCH "zetta.ai/trainig_x0/architectures:architecture_x0"

#EXP_NAME:       "mimic_encodings"
#EXP_VERSION:    "x17"
#TRAINING_ROOT:  "gs://sergiy_exp/training_artifacts"
#LAST_CKPT_PATH: "\(#TRAINING_ROOT)/\(#EXP_NAME)/\(#EXP_VERSION)/last.ckpt"

"@type": "lightning_train"
// use for resuming the WHOLE TRAINING STATE. This will resume training from the checkpoint
// AND DISREGARD PARAMETERS IN THIS FILE, such as new learning rates etc.
// The whole state will be taken from the checkpoint. This is done to maximize reproducibility.
// If you want to load only the weights of the model, it is responsibility of your regime!
// This way, reproducibility is increased
ckpt_path: #LAST_CKPT_PATH
regime: {
	"@type": "NaiveSupervised"
	lr:      4e-4
	model: {
		"@type": "parse_artificery"
		path:    "/mnt/disks/sergiy-x0/code/artificery/params/mimic_emb/embedder_5x5_nonorm.json"
	}
	// model_ckpt_path: #LAST_CKPT_PATH
}
trainer: {
	"@type":            "ZettaDefaultTrainer"
	accelerator:        "gpu"
	devices:            1
	max_epochs:         100
	default_root_dir:   #TRAINING_ROOT
	experiment_name:    #EXP_NAME
	experiment_version: #EXP_VERSION
	log_every_n_steps:  500
	val_check_interval: 500
	checkpointing_kwargs: {
		//update_every_n_secs: 20
		// backup_every_n_secs: 900
	}
	profiler: "simple"
}
train_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  1
	shuffle:     true
	num_workers: 16
	dataset:     DSET.train
}
val_dataloader: {
	"@type":     "TorchDataLoader"
	batch_size:  1
	shuffle:     false
	num_workers: 16
	dataset:     DSET.val
}
