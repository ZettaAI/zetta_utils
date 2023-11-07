#BASE_FOLDER: "gs://zetta-research-dodam/dacey-montaging-research/prototype_7104_320"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [0, 0, 46]
	end_coord: [32768, 32768, 51]
	resolution: [5, 5, 50]

}

#FLOW_TMPL: {
	"@type": "build_subchunkable_apply_flow"
	//expand_bbox_processing: true
	shrink_processing_chunk: false
	skip_intermediaries:     true
	processing_chunk_sizes:  _
	dst_resolution:          _
	op: {
		"@type":         "InterpolateOperation"
		mode:            _
		res_change_mult: _ | *[2, 2, 1]
	}
	bbox: #BBOX
	op_kwargs: {
		src: {
			"@type":    "build_cv_layer"
			path:       _
			read_procs: _ | *[]
		}
	}
	dst: {
		"@type": "build_cv_layer"
		path:    _
	}

}

"@type": "mazepa.execute_locally"
target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		for offset in ["(0,0)", "(0,1)", "(1,0)", "(1,1)"] for res in [10, 20, 40, 80, 160, 320] {
			#FLOW_TMPL & {
				processing_chunk_sizes: [[1024 * 4, 1024 * 4, 1]]

				op: mode: "img"
				op_kwargs: src: path: "\(#BASE_FOLDER)/\(offset)"
				dst: path: "\(#BASE_FOLDER)/\(offset)"
				dst_resolution: [res, res, 50]
			}
		},
	]
}
num_procs: 16
semaphores_spec: {
	read:  8
	write: 8
	cuda:  8
	cpu:   8
}
debug: false
