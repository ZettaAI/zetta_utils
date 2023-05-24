#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [2048, 2048, 10]
	resolution: [512, 512, 45]
}

#FLOW: {
	"@type": "build_subchunkable_apply_flow"
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	processing_chunk_sizes: [[1024, 1024, 1]]
	processing_crop_pads: [[0, 0, 0]]
	expand_bbox_processing: true
	dst_resolution: [256, 256, 45]

	bbox: #BBOX
	src: {
		"@type": "build_constant_volumetric_layer"
		value:   3.14
		read_procs: [
			{
				"@type": "to_uint8"
				"@mode": "partial"
			},
		]
	}
	dst: {
		"@type":             "build_cv_layer"
		path:                "gs://tmp_2w/sergiy/constant_write_x0"
		info_reference_path: "gs://zetta_lee_fly_cns_001_alignment_temp/aced/coarse_x0/raw_img"
		//on_info_exists:      "overwrite"
	}
}

"@type":         "mazepa.execute_on_gcp_with_sqs"
local_test:      true
worker_image:    "..."
worker_replicas: 0
target:          #FLOW
