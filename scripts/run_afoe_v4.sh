#!/bin/bash
# afoe: training-free bank only (V4 = DCT-8 + steerable pyramids), r = 5.
set -e
cd /scratch/haouchat/tfp
A="python -u afoe.py"
$A fit  --name V5   --bank V4 --r 5
$A fit  --name V5c  --chain V5 --r 5
$A eval --chain V5,V5c
$A prox --name V5p  --chain V5,V5c --bank V4 --r 5
$A map  --name V5p  --variant jmono
$A map  --name V5   --variant unf --tune
