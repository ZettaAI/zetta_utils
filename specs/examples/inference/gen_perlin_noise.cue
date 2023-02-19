#DST_PATH: "gs://tmp_2w/examples/perlin"

#CHUNK_SIZE: [1024, 1024, 1]
#DST_INFO_CHUNK_SIZE: [1024, 1024, 1]

#BCUBE: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [1024, 1024, 10]
	resolution: [1, 1, 1]
}

"@type":         "mazepa.execute"
target: {
	"@type": "build_chunked_apply_flow"
	operation: {
		"@type": "VolumetricCallableOperation"
		fn: {
			"@type": "rand_perlin_2d_octaves"
			"@mode": "partial"
			shape: [2, 1024, 1024, 1]
			res: [8, 8]
			octaves: 6
			device: "cuda"
		}
		crop_pad: [0, 0, 0]
	}
	chunker: {
		"@type":      "VolumetricIndexChunker"
		"chunk_size": [1024, 1024, 1]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_chunk_size:     [1024, 1024, 1]
		// info_field_overrides: {
		// 	"data_type": "float32",
		// 	"num_channels": 2,
		// 	"scales": [
		// 		{
		// 			"chunk_sizes": [[1024, 1024, 1]],
		// 			"encoding": "zfpc",
		// 			"key": "1_1_1",
		// 			"resolution": [1, 1, 1],
		// 			"size": [1024, 1024, 10],
		// 			"voxel_offset": [0, 0, 0],
		// 			"zfpc_correlated_dims": [true, true, false, false],
		// 			"zfpc_tolerance": 0.00048828125
		// 		}
		// 	],
		// 	"type": "image"
		// }
	}
	idx: {
		"@type": "VolumetricIndex"
		bbox:    #BCUBE
		resolution: [1, 1, 1]
	}
}