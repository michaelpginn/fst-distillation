#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4GB
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --account=blanca-curc-gpu
#SBATCH --constraint=rome
#SBATCH --out=logs/%j.log
#SBATCH --error=logs/%j.log

set -euo pipefail
set -x

module purge
module load miniforge
# mamba activate distillfst
source activate distillfst
cd "/projects/$USER/fst-distillation"

maxjobs=8
jobcount=0

for lang in aka	ceb	crh	czn	dje	gaa	izh	kon	lin	mao	mlg	nya	ood	orm	ote	san	sot	swa	syc	tgk	tgl	xty	zpv	zul
do
    for order in lex dd
    do
        echo "Running OSTIA-$order for $lang"
        ((jobcount++))
        logname="logs/${jid}_${jobcount}_${lang}_${order}.log"
        srun --exclusive -N1 -n1 python -m src.ostia.run_ostia data/inflection $lang --features --order $order \
              > "$logname" 2>&1 &
        if (( jobcount % maxjobs == 0 )); then
            echo "Reached $maxjobs concurrent jobs — waiting..."
            wait
        fi
    done
done

wait
echo "✅ All runs completed successfully"
