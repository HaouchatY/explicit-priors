#!/bin/bash
# data-analytic PCA (KLT) filters: two closed-form steps, then MAP potentials (r = 3)
cd /scratch/haouchat/tfp
A="python -u afoe.py"
for B in PCA5 PCA7 PCA9; do
  $A fit  --name ${B}r3  --bank $B --r 3
  $A fit  --name ${B}r3c --chain ${B}r3 --r 3
  $A eval --chain ${B}r3,${B}r3c
done
for B in PCA7 PCA9; do
  $A prox --name ${B}r3p --chain ${B}r3,${B}r3c --bank $B --r 3
  $A map  --name ${B}r3p --variant jmono,jpos,mono
done
