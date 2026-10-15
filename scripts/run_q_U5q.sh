#!/bin/bash
set -e
cd /scratch/haouchat/tfp
python -u afoe.py prox --name U5q --chain U5,U5c --bank WCRR60+V4 --r 5
python -u afoe.py map --name U5q --variant jpos,jmono
