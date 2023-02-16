#FIELD_PATH: "gs://zetta_lee_fly_cns_001_alignment_temp/rigid_to_elastic/v1/field"

"@type":      "mazepa.execute_on_gcp_with_sqs"
worker_image: "us.gcr.io/zetta-research/zetta_utils:sergiy_all_p39_x67"
worker_resources: {
	memory: "18560Mi"
}
worker_replicas:     10
batch_gap_sleep_sec: 0.1
local_test:          false
target: {
	"@type":    "lambda"
	lambda_str: "lambda: import time; time.sleep(3600)"
}
