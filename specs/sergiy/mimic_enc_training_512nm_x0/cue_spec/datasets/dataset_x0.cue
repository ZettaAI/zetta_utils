package dataset_x0

let IMG_CV = "https://storage.googleapis.com/fafb_v15_aligned/v0/img/img"
let ENC_CV = "https://storage.googleapis.com/fafb_v15_aligned/v0/emb_fp32/emb_m7"

#dataset_settings: {
	"@type": "LayerDataset"
	layer: {
		"@type": "LayerSet"
		layers: {
			data_in: {
				"@type": "CVLayer"
				path:    IMG_CV
				//cv_kwargs: {cache: true}
				read_postprocs: [
					{
						"@type": "Squeeze"
						dim:     -1
					},
					{
						"@type": "Divide"
						x:       256.0
					},
					{
						"@type": "Add"
						x:       -0.5
					},
				]
			}
			target: {
				"@type": "CVLayer"
				path:    ENC_CV
				//cv_kwargs: {cache: true}
				read_postprocs: [
					{
						"@type": "Squeeze"
						dim:     -1
					},
				]
			}
		}
	}
	sample_indexer: {
		"@type": "VolumetricStepIndexer"
		desired_resolution: [512, 512, 40]
		index_resolution: [512, 512, 40]
		sample_size_resolution: [512, 512, 40]
		sample_size: [512, 512, 1]
		step_size: [128, 128, 1]
		step_size_resolution: [512, 512, 40]
		bcube: {
			"@type":     "BoundingCube"
			start_coord: _
			end_coord:   _
			resolution: [4, 4, 40]
		}
	}
}

train: #dataset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [80000, 30000, 2000]
			end_coord: [230000, 80000, 2099]
			resolution: [4, 4, 40]
		}
	}
}

val: #dataset_settings & {
	sample_indexer: {
		bcube: {
			"@type": "BoundingCube"
			start_coord: [80000, 30000, 2099]
			end_coord: [230000, 80000, 2100]
			resolution: [4, 4, 40]
		}
	}
}
