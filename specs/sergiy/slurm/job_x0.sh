#!/bin/bash
#SBATCH --job-name=testy-test # create a short name for your job
#SBATCH --exclude=sarekl15-5
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --partition=highpri             # Default QOS (the other is preempt)
#SBATCH --cpus-per-task=1               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                # memory per cpu-core (4G is default)
#SBATCH --time=00:00:30                  # total run time limit (HH:MM:SS)

module load anacondapy/2023.03
. ~/.bashrc
conda activate zetta-x1-p310

zetta -vv -l try run -r hypersonic-roaring-teal-of-prestige --no-main-run-process -p -s '{"@type": "mazepa.run_worker"
task_queue: {"@type": "FileQueue", "name": "zzz-hypersonic-roaring-teal-of-prestige-work"}
outcome_queue: {"@type": "FileQueue", "name": "zzz-hypersonic-roaring-teal-of-prestige-outcome", "pull_wait_sec": 2.5}

        sleep_sec: 5
    }'
