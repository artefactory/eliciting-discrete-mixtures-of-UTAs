#!/bin/bash
#SBATCH --job-name=belicit
#SBATCH --output=/gpfs/workdir/auriauvi/honey/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/auriauvi/honey/logs/error_%x_%j.txt
#SBATCH --partition=cpu_long
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL

module purge
module load python/3.9.10/gcc-11.2.0

source /gpfs/users/auriauvi/.venv/guro/bin/activate
cd /gpfs/users/auriauvi/eliciting-discrete-mixtures-of-UTAs/notebooks/hive

python -u bees_elicitation.py --flag b
