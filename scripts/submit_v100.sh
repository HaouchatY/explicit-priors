#!/bin/bash
# usage: submit_sh.sh JOBNAME script.sh   (runs /scratch/haouchat/tfp/script.sh on one GPU)
export SUPPRESS_DEPRECATION_MESSAGE=true
NAME=$1; SCRIPT=$2
runai submit --name "$NAME" --image registry.rcp.epfl.ch/haouchat/llvm2 --gpu 1 \
  --node-pools v100 --backoff-limit 0 \
  --existing-pvc claimname=lib-scratch,path=/scratch --supplemental-groups 82403 \
  -e PYTHONUSERBASE=/scratch/haouchat/.local -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  --command -- bash -lc "bash /scratch/haouchat/tfp/$SCRIPT >> /scratch/haouchat/tfp/logs/$NAME.log 2>&1" 2>&1 | tail -1
