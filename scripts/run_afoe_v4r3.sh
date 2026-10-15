#!/bin/bash
# training-free bank at r = 3 (the window of the best WCRR60 projection)
set -e
cd /scratch/haouchat/tfp
A="python -u afoe.py"
$A fit  --name V3   --bank V4 --r 3
$A fit  --name V3c  --chain V3 --r 3
$A eval --chain V3,V3c
$A prox --name V3p  --chain V3,V3c --bank V4 --r 3
$A map  --name V3p  --variant jmono,mono
$A check --name V3p --variant jmono
