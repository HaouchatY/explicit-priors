#!/bin/bash
# afoe validation on WCRR's 60 filters (r = 3): reproduce the tfp numbers.
set -e
cd /scratch/haouchat/tfp
A="python -u afoe.py"
$A fit  --name W60r3  --bank WCRR60 --r 3
$A fit  --name W60r3c --chain W60r3 --r 3
$A eval --chain W60r3,W60r3c
$A prox --name W60r3p --chain W60r3,W60r3c --bank WCRR60 --r 3
$A map  --name W60r3p --variant jmono,mono
$A map  --name W60r3p --variant jmono,mono --tune
$A map  --name W60r3  --variant unf --tune
