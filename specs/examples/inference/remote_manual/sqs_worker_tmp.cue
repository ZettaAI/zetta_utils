"@type": "mazepa.run_worker"
exec_queue: {
	"@type":        "mazepa.SQSExecutionQueue"
	name:           "zzz-exec-real-wealthy-tapir-of-infinity-work"
	pull_lease_sec: 30
}
max_pull_num: 3
sleep_sec:    8
