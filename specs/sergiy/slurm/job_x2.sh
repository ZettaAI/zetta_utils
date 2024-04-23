#!/bin/sh

#SBATCH --job-name            slurm-worker-quiet-peculiar-inchworm-of-fragrance
#SBATCH --ntasks              25
#SBATCH --cpus-per-task       1
#SBATCH --mem-per-cpu         8G
#SBATCH --partition           highpri
#SBATCH --time                0-00:10:04
#SBATCH --exclude             sarekl15-5



module purge
module load  anacondapy/2023.03
conda activate zetta-x1-p310
zetta -vv -l try run -r quiet-peculiar-inchworm-of-fragrance --no-main-run-process -p -s '{"@type": "mazepa.run_worker"
task_queue: {"@type": "FileQueue", "name": "zzz-quiet-peculiar-inchworm-of-fragrance-work"}
outcome_queue: {"@type": "FileQueue", "name": "zzz-quiet-peculiar-inchworm-of-fragrance-outcome", "pull_wait_sec": 2.5}

        sleep_sec: 5
    }'
