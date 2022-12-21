IMAGE=us.gcr.io/zetta-research/zetta_utils:inference_x10
QUEUE_NAME='"zzz-exec-real-wealthy-tapir-of-infinity-work"'

docker run -v ${PWD}:/tmp/  --env-file worker_env -it $IMAGE  zetta -l try -vv run -s '{
"@type": "mazepa.run_worker"
exec_queue: {
"@type": "mazepa.SQSExecutionQueue"
name: '$QUEUE_NAME'
pull_lease_sec: 10
}
}'
docker run -v ${PWD}:/tmp/  --env-file worker_env -it $IMAGE python --version
