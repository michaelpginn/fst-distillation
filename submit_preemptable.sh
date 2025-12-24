#!/bin/bash
set -x

# Require at least 1 argument (cluster)
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [args...]"
    echo "  args... : all remaining args passed to sweep_job.sh"
    exit 1
fi

sbatch \
  --qos="preemptable" \
  --account="blanca-blast-lecs" \
  --gres=gpu:1 \
  sweep_job.sh  --models /scratch/alpine/$USER/fst-distillation/models/ "$@"
