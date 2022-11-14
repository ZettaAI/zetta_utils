"@type": "run_mazepa_worker"
exec_queue: {
	"@type":            "MazepaSQSExecutionQueue"
	name:               "sergiy-corgie-x0"
	outcome_queue_name: "sergiy-corgie-completion-x0"
	pull_lease_sec:     30
}
max_pull_num: 3
sleep_sec:    8
