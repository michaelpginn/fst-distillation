#!/bin/bash
set -x

# Require at least 1 argument (cluster)
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <cluster> [args...]"
    echo "  cluster : clearlab1 | curc-gpu | blast-lecs"
    echo "  args... : all remaining args passed to sweep_job.sh"
    exit 1
fi

cluster="$1"
shift 1   # everything else goes to sweep_job.sh

case "$cluster" in
  clearlab1)
    qos="blanca-clearlab1"
    part="blanca-clearlab1"
    acct="blanca-clearlab1"
    ;;
  curc-gpu)
    qos="blanca-curc-gpu"
    part="blanca-curc-gpu"
    acct="blanca-curc-gpu"
    ;;
  blast-lecs)
    qos="blanca-blast-lecs"
    part="blanca-blast-lecs"
    acct="blanca-blast-lecs"
    ;;
  *)
    echo "Unknown cluster: $cluster"
    exit 1
    ;;
esac

sbatch \
  --qos="$qos" \
  --partition="$part" \
  --account="$acct" \
  --gres=gpu:1 \
  sweep_job.sh  --models /scratch/alpine/$USER/fst-distillation/models/ "$@"
