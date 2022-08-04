package dataset_x0

let IMG_CV = "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
let ENC_CV = "https://storage.googleapis.com/fafb_v15_aligned/v0/experiments/emb_fp32/baseline_downs_emb_m2_m4_x0"

"<type>": "LayerDataset"
layer: {
	"<type>": "LayerSet"
	layers: {
		data_in: {
			"<type>": "CVLayer"
			path:     IMG_CV
			read_postprocs: [
				{
					"<type>": "Squeeze"
					dim:      -1
				},
				{
					"<type>": "Divide"
					x:        256.0
				},
				{
					"<type>": "Add"
					x:        -0.5
				},
			]
		}
		target: {
			"<type>": "CVLayer"
			path:     ENC_CV
			read_postprocs: [
				{
					"<type>": "Squeeze"
					dim:      -1
				},
			]
		}
	}
}
sample_indexer: {
	"<type>": "VolumetricStepIndexer"
	desired_resolution: [64, 64, 40]
	index_resolution: [64, 64, 40]
	sample_size_resolution: [64, 64, 40]
	sample_size: [1024, 1024, 1]
	bcube: {
		"<type>": "BoundingCube"
		start_coord: [70000, 20000, 2000]
		end_coord: [240000, 90000, 2100]
		resolution: [4, 4, 40]
	}
	step_size: [256, 256, 1]
	step_size_resolution: [64, 64, 40]
}
