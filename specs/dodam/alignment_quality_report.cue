#SRC_PATH:   "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v3/misalignment/misalignment_cns_img_m5"

#XY_FOV: 512
#Z_FOV: 5
#BCUBE_RESOLUTION: [4, 4, 45]
#MISALIGNMENT_THRESHOLDS: [0.5, 1.0, 1.5, 2.0, 2.5]
#NUM_WORST_CHUNKS: 5

#IDX: {
	"@type": "VolumetricIndex"
	bcube: {
		"@type": "BoundingCube"
		start_coord: [1024 * 128, 1024 * 48, 2950]
		end_coord: [1024 * 160, 1024 * 64, 3050]
		resolution: #BCUBE_RESOLUTION
	}
	resolution: [128, 128, 45]
}

"@type": "mazepa_execute"
target: {
	"@type": "compute_alignment_quality"
	chunker: {
		"@type": "VolumetricIndexChunker"
		"chunk_size": [#XY_FOV, #XY_FOV, #Z_FOV]
		"step_size": [#XY_FOV, #XY_FOV, #Z_FOV]
	}
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
        resolution: #BCUBE_RESOLUTION
        misalignment_thresholds: #MISALIGNMENT_THRESHOLDS
	idx: #IDX
        num_worst_chunks: #NUM_WORST_CHUNKS
}
