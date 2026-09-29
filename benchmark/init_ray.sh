#!/bin/bash

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/anaconda3/bin:$PATH"
fi
HEAD_NODE_IP=$1
PORT=$2
RANK=$3


if [[ $RANK -eq 0 ]]; then
    echo 'Start ray head'
    ray start --head --port=${PORT}
else
    echo 'Add ray node'
    ray start --address=${HEAD_NODE_IP}:${PORT}
fi

sleep 5

ray status