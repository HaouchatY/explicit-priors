#!/bin/bash
set -e
cd /scratch/haouchat/tfp
python -u afoe.py prox --name W60r3q --chain W60r3,W60r3c --bank WCRR60 --r 3
python -u afoe.py map --name W60r3q --variant jpos,jmono
