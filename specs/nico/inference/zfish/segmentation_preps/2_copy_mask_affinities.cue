#AFF_SRC_PATH:      "gs://zetta_jlichtman_zebrafish_001_kisuk/final/v2/affinity"
#AFF_DST_PATH:      "gs://zetta_jlichtman_zebrafish_001_kisuk/final/v2/affinity_resin_mask"
#BLACKOUT_MSK_PATH: "gs://zetta_jlichtman_zebrafish_001_alignment_temp/fine_full_v2/masks/img_close9_open5"

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-jlichtman-zebrafish-001/zetta_utils:nico_py3.9_20230630"
worker_cluster_name:    "zutils-zfish"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-jlichtman-zebrafish-001"
worker_resources: {
	memory: "27560Mi"
}
worker_replicas: 100
local_test:      false

#AFF_SRC: {
	"@type": "build_cv_layer"
	path:    #AFF_SRC_PATH
}

#AFF_DST: {
	"@type":             "build_cv_layer"
	path:                string
	info_reference_path: #AFF_SRC_PATH
	info_chunk_size: [256, 256, 128]
	cv_kwargs: {"delete_black_uploads": false} // ws+agg require all affinities present, so cannot issue DELETES
}

#BLACKOUT_MASK: {
	"@type": "build_cv_layer"
	path:    #BLACKOUT_MSK_PATH
	data_resolution: [128, 128, 30]
	interpolation_mode: "mask"
}

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [1024, 1024, 0]
	end_coord: [10240, 15360, 4010]
	resolution: [32, 32, 30]
}

target: {
	"@type": "mazepa.concurrent_flow"
	stages: [
		for factor in [0.0] {
			"@type": "build_subchunkable_apply_flow"
			// "@mode": "partial"
			bbox: #BBOX
			dst:  #AFF_DST & {path: #AFF_DST_PATH}
			dst_resolution: [16, 16, 30]
			processing_chunk_sizes: [[1024, 1024, 256]]
			processing_crop_pads: [[0, 0, 0]]
			processing_blend_pads: [[0, 0, 0]]
            expand_bbox: true
			op: {
				"@type": "VolumetricCallableOperation"
				fn: {
					"@type":      "lambda"
					"lambda_str": "lambda src: src['aff'] * src['mask'].clamp(0.0,1.0)"
				}
			}
			op_kwargs: {
				src: {
					"@type": "build_layer_set"
					layers: {
						aff:  #AFF_SRC
						mask: #BLACKOUT_MASK
					}
				}
			}
		},
	]
}
