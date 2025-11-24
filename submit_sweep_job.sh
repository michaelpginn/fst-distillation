#!/bin/bash

set -x

no_override=0
while [[ "$1" == --* ]]; do
    case "$1" in
        --no-override)
            no_override=1
            shift
            ;;
        *)
            echo "Unknown flag: $1"
            exit 1
            ;;
    esac
done

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 [--no-override] <cluster> <lang> <objective>"
    echo "  cluster   : clearlab1 | curc-gpu | blast-lecs"
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
  *)
    echo "Unknown cluster: $cluster"
    echo "Expected: clearlab1, curc-gpu, blast-lecs"
    exit 1
    ;;
esac

if [[ "$no_override" == 1 ]]; then
    override=""
else
    override="--override-alignment"
fi

sbatch \
  --qos="$qos" \
  --partition="$part" \
  --account="$acct" \
  --gres=gpu:1 \
  sweep_job.sh data/inflection "$lang" --features \
    --objective "$objective" \
    $override \
    --models /scratch/alpine/$USER/fst-distillation/models/
