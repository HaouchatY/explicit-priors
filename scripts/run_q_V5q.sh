#!/bin/bash
set -e
cd /scratch/haouchat/tfp
python -u afoe.py prox --name V5q --chain V5,V5c --bank V4 --r 5
python -u afoe.py map --name V5q --variant jpos,jmono
