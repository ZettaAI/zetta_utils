import "list"

#SRC_PATH: "tigerdata://sseung-test1/ca3-alignment-temp/full_section_imap4"
#DST_PATH: "tigerdata://sseung-test1/sergiy-temp/tmp_x0"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 0]
	end_coord: [1024 * 100, 1024 * 100, 10]
	resolution: [4, 4, 45]
}

"@type":                "mazepa.execute_on_slurm"
worker_replicas: 1
worker_resources: {
	cpus_per_task: 4,
	mem_per_cpu: "8G"
	// sres: ...
	array: list.Range(0,4)
}
init_command:  "module load  anacondapy/2024.02; conda activate zetta-x1-p310"
local_test:    false // set to `false` execute remotely
message_queue: "sqs"

target: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX
	expand_bbox_resolution: true

	dst_resolution: [384, 384, 45]

	processing_chunk_sizes: [[2 * 1024, 2 * 1024, 1]]
	processing_crop_pads: [[0, 0, 0]]
	processing_blend_pads: [[0, 0, 0]]
	skip_intermediaries: true

	expand_bbox_processing: true

	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
		}
	}

	dst: {
		"@type":             "build_cv_layer"
		path:                #DST_PATH
		info_reference_path: #SRC_PATH
	}
}
