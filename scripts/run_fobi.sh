#!/bin/bash
# closed-form ICA (FOBI) filters: two closed-form steps (r = 3)
cd /scratch/haouchat/tfp
A="python -u afoe.py"
for B in FOBI5 FOBI7 FOBI9; do
  $A fit  --name ${B}r3  --bank $B --r 3
  $A fit  --name ${B}r3c --chain ${B}r3 --r 3
  $A eval --chain ${B}r3,${B}r3c
done
