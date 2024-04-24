#IMG_PATH: "matrix://ca3-alignment-2024-tmp/coarse/img"
#DST_FOLDER: "matrix://ca3-alignment-2024-tmp/sergiy_tests/aio_x0"

#ROI_BOUNDS: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [102400, 96256, 6112]
	resolution: [1, 1, 1]
}
#BASE_RES: [4, 4, 42]

#TEST_LOCAL: true
#CLUSTER_NUM_WORKERS: 16

#RUN_ENCODE: true
#RUN_DEFECT: true
#RUN_MASK_ENCODE: true
#RUN_ALIGN: true
#RUN_MISD: true



#PAIR_FLOW_TMPL: {
    "@type": "PairwiseAlightmentFlowSchema"
    dst_folder: "gs://dkronauer-ant-001-alignment/\(#EXP_NAME)"
    defect_model_path: #DEFECT_MODEL_PATH
}


"@type":         "mazepa.execute_on_slurm"
worker_replicas: #NUM_WORKERS
worker_resources: {
	cpus_per_task: 8
	mem_per_cpu:   "4G"
	gres: ["gpu:3090:8"]
}
init_command:  "module load anacondapy/2023.03; conda activate zetta-x1-p310"
message_queue: "sqs"
worker_replicas:     
local_test:       #TEST_LOCAL
target:           #TOP_LEVEL_FLOW 

#DEFECT_MODEL_PATH: "gs://zetta_lee_fly_cns_001_models/jit/20221114-defects-step50000.static-1.11.0.jit"

#ENC_MODEL_PATHS: [
    {
        path: "gs://zetta-research-nico/training_artifacts/general_encoder_loss/4.0.1_M3_M3_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.03_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [1, 1, 1] 
    },
    {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M4_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.06_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [2, 2, 1] 
    },
    {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M5_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.08_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [4, 4, 1] 
    },
    {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.4.0_M3_M6_C1_lr0.0002_locality1.0_similarity0.0_l10.05-0.12_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [8, 8, 1] 
    },
    {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/4.0.0_M3_M7_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [16, 16, 1] 
    },
    {
        path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M3_M8_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
        res_change_mult: [32, 32, 1] 
    },
    {
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.2_M4_M9_C1_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M5_M10_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.1_M6_M11_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
	{
		path: "gs://zetta-research-nico/training_artifacts/general_coarsener_loss/1.0.0_M7_M12_C4_lr0.0002_locality1.0_similarity0.0_l10.0-0.12_N1x4/last.ckpt.model.spec.json"
		res_change_mult: [32, 32, 1]
	},
]
