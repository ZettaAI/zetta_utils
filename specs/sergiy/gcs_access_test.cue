"@type":         "mazepa.execute_on_gcp_with_sqs"
worker_image:    "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x69"
worker_replicas: 1
worker_resources: {}
local_test: false

target: {
	"@type":    "test_gcs_access"
	"@mode":    "partial"
	read_path:  "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field/info"
	write_path: "gs://zetta_lee_fly_cns_001_alignment_temp/tmp/dummy_x0"
}
