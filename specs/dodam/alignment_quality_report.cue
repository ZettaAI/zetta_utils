#SRC_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/fine_v4/misalignment/misalignment_cns_img_m5"
#DATA_RESOLUTION: [128, 128, 45]
#XY_FOV: 512
#Z_FOV:  5

#BCUBE_RESOLUTION: [4, 4, 45]
#BCUBE_START: [1024 * 128, 1024 * 32, 2990]
#BCUBE_END: [1024 * 160, 1024 * 48, 3010]
#MISALIGNMENT_THRESHOLDS: [0.5, 1.0, 1.5, 2.0, 2.5]
#NUM_WORST_CHUNKS: 5

#IDX: {
	"@type": "VolumetricIndex"
	bbox: {
		"@type":     "BBox3D.from_coords"
		start_coord: #BCUBE_START
		end_coord:   #BCUBE_END
		resolution:  #BCUBE_RESOLUTION
	}
	resolution: #DATA_RESOLUTION
}

"@type": "mazepa.execute"
target: {
	"@type": "compute_alignment_quality"
	chunk_size: [#XY_FOV, #XY_FOV, #Z_FOV]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	resolution:              #BCUBE_RESOLUTION
	misalignment_thresholds: #MISALIGNMENT_THRESHOLDS
	idx:                     #IDX
	num_worst_chunks:        #NUM_WORST_CHUNKS
}
