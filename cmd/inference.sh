#!/bin/bash

# usage ----------------------------------------------------------
# bash cmd/inference.sh exps/cifar.yaml
# ----------------------------------------------------------------
python3 src/acmap/inference.py --config $1 \
                      --init_cls 0 \
                      --increment 5 \
                      --seed 1993 1994 1995 1996 1997
