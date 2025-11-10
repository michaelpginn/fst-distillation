#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --constraint=rome
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

module purge
module load miniforge
mamba activate distillfst
cd "/projects/$USER/fst-distillation"

set -x


for lang in aka	ceb	crh	czn	dje	gaa	izh	kon	lin	mao	mlg	nya	ood	orm	ote	san	sot	swa	syc	tgk	tgl	xty	zpv	zul
do
for order in lex dd
do
echo Running OSTIA-$order for $lang
srun python -m src.ostia.run_ostia data/inflection $lang --features --order $order
done
done
