"@type": "mazepa.k8s.configure_cronjob"
cluster: {
	"@type": "mazepa.k8s.ClusterInfo"
	name: "zutils-x3"
	region: "us-east1"
	project: "zetta-research"
}
name: "gc-cron"
namespace: "default"
image: "us.gcr.io/zetta-research/zetta_utils:gc_cronjob_v2"
command: ["/bin/sh"]
command_args: [
	"-c",
	"python -m zetta_utils.cloud_management.resource_cleanup"
]
env_vars: {
	"EXECUTION_HEARTBEAT_LOOKBACK": "300",
}
preset_env_vars: [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION"
]
resources: {
	memory: "500Mi"
}
spec_config: {
	"@type": "mazepa.k8s.CronJobSpec"
	schedule: "*/15 * * * *"
}
patch: true
