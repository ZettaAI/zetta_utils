import "list"

#SRC_IMG_PATH: "gs://dacey-human-retina-001-drop/rough_alignment/2310_trakem"
#DST_IMG_PATH: "gs://dacey-human-retina-001-drop/rough_alignment/2310_trakem"

#Z_START: 1
#Z_END: 3030

#BBOX: {
	"@type": "BBox3D.from_coords"
	start_coord: [int, int, int] | *[0, 0, 1]
	end_coord: [int, int, int] | *[42989, 43567, 3030]
	resolution: [5, 5, 50]
}


"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/dacey-human-retina-001/zetta_utils:nico_py3.9_20231109"
worker_resources: {
	memory:           "13560Mi"
}
worker_replicas:      300
batch_gap_sleep_sec:  0.1
do_dryrun_estimation: true
local_test:           false
worker_cluster_project: "dacey-human-retina-001"
worker_cluster_region: "us-east1"
worker_cluster_name: "zutils"

target: {
	"@type": "mazepa.sequential_flow"
	stages: [
		for res in [5, 10, 20, 40, 80, 160] {
			"@type":        "build_interpolate_flow"
			mode:           "img"
			src_resolution: [res, res, 50]
			dst_resolution: [res * 2, res * 2, 50]

			chunk_size: [2048,2048,64]

			bbox: #BBOX
			src: {
				"@type": "build_cv_layer"
				path:    #SRC_IMG_PATH
			}
			dst: {
				"@type": "build_cv_layer"
				path:    #DST_IMG_PATH
				cv_kwargs: {"compress": false}
			}
		},
	]
}
