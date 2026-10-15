#!/bin/bash
# same pipeline for PCA13 and the other filter families (r = 3)
cd /scratch/haouchat/tfp
A="python -u afoe.py"
for B in PCA13 DCT8 ICA; do
  $A fit  --name ${B}r3  --bank $B --r 3
  $A fit  --name ${B}r3c --chain ${B}r3 --r 3
  $A eval --chain ${B}r3,${B}r3c
done
