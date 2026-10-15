#!/bin/bash
# afoe final: union bank (WCRR60 + V4), r = 5; projections onto the union and onto WCRR's 60 filters.
set -e
cd /scratch/haouchat/tfp
A="python -u afoe.py"
$A fit  --name U5   --bank WCRR60+V4 --r 5
$A fit  --name U5c  --chain U5 --r 5
$A eval --chain U5,U5c
$A prox --name W60pU --chain U5,U5c --bank WCRR60 --r 5
$A map  --name W60pU --variant jmono,mono
$A prox --name U5p  --chain U5,U5c --bank WCRR60+V4 --r 5
$A map  --name U5p  --variant jmono
$A map  --name W60pU --variant jmono,mono --tune
$A map  --name U5p  --variant jmono --tune
$A map  --name U5   --variant unf --tune
