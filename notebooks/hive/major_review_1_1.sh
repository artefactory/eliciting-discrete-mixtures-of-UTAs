#!/bin/bash
#SBATCH --job-name=run_1
#SBATCH --output=/gpfs/workdir/auriauvi/mr_1/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/auriauvi/mr_1/logs/error_%x_%j.txt
#SBATCH --partition=cpu_long
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL

module purge
module load python/3.9.10/gcc-11.2.0

source /gpfs/users/auriauvi/.venv/guro/bin/activate
cd /gpfs/users/auriauvi/eliciting-discrete-mixtures-of-UTAs/notebooks/hive

python -u radius_xp.py --flag major_review_1
