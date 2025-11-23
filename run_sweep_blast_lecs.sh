#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=10
#SBATCH --time=2-00:00:00
#SBATCH --qos=blanca-blast-lecs
#SBATCH --partition=blanca-blast-lecs
#SBATCH --account=blanca-blast-lecs
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge
mamba activate distillfst
cd "/projects/$USER/fst-distillation"

set -x

echo Running for $1 with objective $2
python -m src.sweep data/inflection $1 --features --objective $2 --models /scratch/alpine/$USER/fst-distillation/models/ --override-alignment
