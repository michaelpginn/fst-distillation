#!/bin/bash
#
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <cluster> <lang> <objective>"
    echo "  cluster   : clearlab1 | curc-gpu | blast-lecs | curc"
    echo "  lang      : language code (e.g., ceb, lin, tgk)"
    echo "  objective : your sweep objective"
    exit 1
fi

cluster="$1"
lang="$2"
objective="$3"

case "$cluster" in
  clearlab1)
    qos="blanca-clearlab1"
    part="blanca-clearlab1"
    acct="blanca-clearlab1"
    need_gpu=1
    ;;
  curc-gpu)
    qos="blanca-curc-gpu"
    part="blanca-curc-gpu"
    acct="blanca-curc-gpu"
    need_gpu=1
    ;;
  blast-lecs)
    qos="blanca-blast-lecs"
    part="blanca-blast-lecs"
    acct="blanca-blast-lecs"
    need_gpu=1
    ;;
  curc)
    qos="blanca-curc"
    part="blanca-curc"
    acct="blanca-curc"
    need_gpu=0
    ;;
  *)
    echo "Unknown cluster: $cluster"
    echo "Expected: clearlab1, curc-gpu, blast-lecs, curc"
    exit 1
    ;;
esac

if [[ "$need_gpu" == 1 ]]; then
    gres="--gres=gpu:1"
    override="--override-alignment"
else
    gres=""
    override="--override-alignment"
fi

sbatch \
  --qos="$qos" \
  --partition="$part" \
  --account="$acct" \
  $gres \
  job.sh data/inflection "$lang" --objective "$objective" "$override" --models /scratch/alpine/$USER/fst-distillation/models/
