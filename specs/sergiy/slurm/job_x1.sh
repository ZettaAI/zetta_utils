#!/bin/sh

#SBATCH --nodes               10
#SBATCH --time                0-00:10:04
#SBATCH --exclude             sarekl15-5
#SBATCH --ntasks              1
#SBATCH --partition           highpri
#SBATCH --job-name            testy-test

module purge
module load  anacondapy/2023.03
conda activate zetta-x1-p310
zetta -vv -l try run -r daring-ubiquitous-wren-of-penetration --no-main-run-process -p -s '{"@type": "mazepa.run_worker"
task_queue: {"@type": "FileQueue", "name": "zzz-daring-ubiquitous-wren-of-penetration-work"}
outcome_queue: {"@type": "FileQueue", "name": "zzz-daring-ubiquitous-wren-of-penetration-outcome", "pull_wait_sec": 2.5}

        sleep_sec: 5
    }'
