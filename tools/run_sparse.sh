#!/usr/bin/env bash

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3,4,5,6,7" \
    tools/sparse_train.py \
        -c ./configs/MobileNetV1/MobileNetV1.yaml \
        -o print_interval=10 \
        -t 12 \
        -g 16
