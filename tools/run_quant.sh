#!/usr/bin/env bash

python3.7 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/quant_train.py \
        -c ./configs/MobileNetV2/MobileNetV2.yaml \
        -o print_interval=10 \
        -o pretrained_model="./output/MobileNetV2/best_model/ppcls"\
        -o epochs=40 \
        -o LEARNING_RATE.params.lr=0.0045 \
        -t 12 \
        -g 16
