#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [2 * 1024, 2 * 1024, 363]
	end_coord: [11 * 1024, 11 * 1024, 364]
	resolution: [32, 32, 42]
}
local_test:      true
debug:           true
worker_replicas: 16

#PAIR_FLOW: {
	"@type":           "build_pairwise_align_flow"
	img_path:          "gs://dkronauer-ant-001-raw/brain"
	dst_dir:        "gs://sergiy_exp/aio_test_x0"
	defect_model_path: "gs://zetta_lee_fly_cns_001_models/jit/20221114-defects-step50000.static-1.11.0.jit"
	base_resolution: [4, 4, 42]
	scale_specs:         #ENC_MODELS
	bbox:                #ROI_BOUNDS
	skip_enc:            false
	skip_defect_detect:  true
	skip_defect_bin:     true
	on_info_exists_mode: "overwrite"
	// defect_binarization_fn: {
	//     "@type": "lambda",
	//     "lambda_str": "lambda src: (src > 0).byte()"
	// }
}

"@type": "mazepa.execute_on_slurm"
worker_resources: {
	cpus_per_task: 8
	mem_per_cpu:   "4G"
	gres: ["gpu:3090:8"]
}
init_command:  "module load anacondapy/2023.03; conda activate zetta-x1-p310"
message_queue: "sqs"
target:        #PAIR_FLOW

#SCALE_SPECS: [
	{
		enc_params: {
			model_path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
			res_change_mult: [1, 1, 1]
		}
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
			sm:      25, num_iter: 200, lr: 0.100
		}
	},
	{
		enc_params: {
			model_path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
			res_change_mult: [2, 2, 1]
		}
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
            sm: 50, num_iter: 300, lr: 0.100
        }
	},
	{
		enc_params: {
			model_path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
			res_change_mult: [4, 4, 1]
		}
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
            sm: 100, num_iter: 500, lr: 0.050}
	},
	{
		enc_params: {
			model_path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
			res_change_mult: [8, 8, 1]
		}
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
            sm: 150, num_iter: 700, lr: 0.030}
	},
	{
		enc_params: {
			model_path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
			res_change_mult: [16, 16, 1]
		}
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
            sm: 200, num_iter: 700, lr: 0.030}
	},
	{
		enc_params: {
			model_path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M3_M8_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
			res_change_mult: [32, 32, 1]
		}
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
            sm: 300, num_iter: 700, lr: 0.015}
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.2_M4_M9_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
		cf_fn: {
			"@type": "align_with_online_finetuner"
			"@mode": "partial"
            sm: 300, num_iter: 700, lr: 0.015}
	},
]
