#AFF_PATH:         "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity"
#BACKUP_PATH:      "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity/backup_for_adjustments"
#COPY_PATH:        "gs://zetta_lee_fly_cns_001_kisuk/final/v2/affinity/cutouts/test002"
#AFF_COPY_PATH:    #COPY_PATH + "/affinity"
#BACKUP_COPY_PATH: #COPY_PATH + "/backup"

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [1602, 306, 2834]
	end_coord: [1602 + 18, 306 + 18, 2834 + 13]
	resolution: [256, 256, 45]
}

"@type":                "mazepa.execute_on_gcp_with_sqs"
worker_image:           "us.gcr.io/zetta-lee-fly-vnc-001/zetta_utils:tmacrina_mask_affinities_x3"
worker_cluster_name:    "zutils-cns"
worker_cluster_region:  "us-east1"
worker_cluster_project: "zetta-lee-fly-vnc-001"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas: 1
local_test:      true

#COPY_TEMPLATE: {
	"@type": "build_subchunkable_apply_flow"
	bbox:    #BBOX
	dst_resolution: [16, 16, 45]
	processing_chunk_sizes: [[144 * 2, 144 * 2, 13]]
	processing_crop_pads: [[0, 0, 0]]
	processing_blend_pads: [[0, 0, 0]]
	expand_bbox_processing: true

	fn: {
		"@type":    "lambda"
		lambda_str: "lambda src: src"
	}
	op_kwargs: _
	dst:       _
}

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		#COPY_TEMPLATE & {
			op_kwargs: {
				src: {
					"@type": "build_cv_layer"
					path:    #AFF_PATH
				}
			}
			dst: {
				"@type":             "build_cv_layer"
				path:                #AFF_COPY_PATH
				info_reference_path: #AFF_PATH
				cv_kwargs: {"delete_black_uploads": false}
			}
		},

		#COPY_TEMPLATE & {
			op_kwargs: {
				src: {
					"@type": "build_cv_layer"
					path:    #BACKUP_PATH
				}
			}
			dst: {
				"@type":             "build_cv_layer"
				path:                #BACKUP_COPY_PATH
				info_reference_path: #BACKUP_PATH
				cv_kwargs: {"delete_black_uploads": false}
			}
		},

	]

}
