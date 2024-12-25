#!/bin/bash

# usage ----------------------------------------------------------
# nohup run/nohup.sh train exps/cifar.yaml &
# nohup run/nohup.sh inference exps/cifar.yaml &
# ----------------------------------------------------------------
if [[ $1 == "train" ]]; then
    bash docker.sh exec "cmd/train.sh $2"
elif [[ $1 == "inference" ]]; then
    bash docker.sh exec "cmd/inference.sh $2"
else
    echo "usage: bash nohup.sh [train|inference] [config]"
fi
