import "strconv"

//
// Handy variables
#SRC_PATH: "gs://sergiy_exp/hippo/missing/cutout_x0/img"

#MIDDLE: 1266
#BBOX_TMPL: {
	"@type": "BBox3D.from_coords"
	start_coord: [36 * 1024, 42 * 1024, int]
	end_coord: [37 * 1024, 43 * 1024, int]
	resolution: [24, 24, 45]
}

#COPY_MAP: {
	"1266": 1267,
	"1265": 1267,
	"1264": 1267,
	"1263": 1261,
	"1262": 1261,
}
// Execution parameters
"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p310_x214"
worker_cluster_name:    "zutils-x3"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-research"
worker_resources: {
	memory: "18560Mi"
	//"nvidia.com/gpu": "1"
}
worker_replicas: 10
debug: true
local_test:      true // set to `false` execute remotely

#COPY_TMPL: {
	// We're applying subchunkable processing flow,
	"@type": "build_subchunkable_apply_flow"
	bbox:   #BBOX_TMPL 

	// What resolution is our destination?
	dst_resolution: [24, 24, 45]

	// How do we chunk/crop/blend?
	processing_chunk_sizes: [[1024, 1024, 16]]
	processing_crop_pads: [[0, 0, 0]]
	processing_blend_pads: [[0, 0, 0]]
	skip_intermediaries: true

	// We want to expand the input bbox to be evenly divisible
	// by chunk size
	expand_bbox_processing: true

	// Specification for the operation we're performing
	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	// Specification for the inputs to the operation
	op_kwargs: {
		src: {
			"@type": "build_cv_layer"
			path:    #SRC_PATH
			index_procs: [
				{
					"@type": "VolumetricIndexTranslator"
					offset: [0, 0, _]
					resolution: [24, 24, 45]
				},
			]
		}
	}

	// Specification of the output layer. Subchunkable expects
	// a single output layer. If multiple output layers are
	// needed, refer to advanced examples.
	dst: {
		"@type":             "build_cv_layer"
		path:                #SRC_PATH
	}
}

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for tgt_z, src_z in #COPY_MAP
		{
			#COPY_TMPL & {
				bbox: {
					start_coord: [_, _, strconv.Atoi(tgt_z)]
					end_coord: [_, _, strconv.Atoi(tgt_z) + 1]
				}	
				op_kwargs: src: index_procs: [{offset: [_, _, src_z - strconv.Atoi(tgt_z)]}]
			}
		}

	]
}