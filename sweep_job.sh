#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=16
#SBATCH --time=7-00:00:00
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge
mamba activate distillfst
cd "/projects/$USER/fst-distillation"

set -x

python -m src.sweep "$@"
