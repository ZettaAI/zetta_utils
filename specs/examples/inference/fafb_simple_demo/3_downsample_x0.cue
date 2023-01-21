#SRC_PATH: "gs://tmp_2w/inference_tests/raw_img_x20"

"@type": "mazepa.execute"
target: {
	"@type": "build_interpolate_flow"
	bbox: {
		"@type": "BBox3D.from_coords"
		start_coord: [1024 * 2, 1024 * 0, 2000]
		end_coord: [1024 * 10, 1024 * 8, 2005]
		resolution: [64, 64, 40]
	}
	dst_resolution: [128, 128, 40]
	src_resolution: [64, 64, 40]
	chunk_size: [2 * 1024, 2 * 1024, 1]
	src: {
		"@type": "build_cv_layer"
		path:    #SRC_PATH
	}
	mode: "img"
}
