# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import paddle

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model, save_model
from ppcls.utils import logger
from paddleslim.dygraph.quant import QAT
import program


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/ResNet/ResNet50.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    parser.add_argument(
        '-t',
        '--target_rate',
        type=int,
        default='0',
        help='remained weights per group')
    parser.add_argument(
        '-g',
        '--group_size',
        type=int,
        default='16',
        help='remained weights per group')
    args = parser.parse_args()
    return args

quant_config = {
        # weight preprocess type, default is None and no preprocessing is performed. 
        'weight_preprocess_type': None,
        # activation preprocess type, default is None and no preprocessing is performed.
        'activation_preprocess_type': 'PACT',
        # weight quantize type, default is 'channel_wise_abs_max'
        'weight_quantize_type': 'channel_wise_abs_max',
        # activation quantize type, default is 'moving_average_abs_max'
        'activation_quantize_type': 'moving_average_abs_max',
        # weight quantize bit num, default is 8
        'weight_bits': 8,
        # activation quantize bit num, default is 8
        'activation_bits': 8,
        # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
        'dtype': 'int8',
        # window size for 'range_abs_max' quantization. default is 10000
        'window_size': 10000,
        # The decay coefficient of moving average, default is 0.9
        'moving_rate': 0.9,
        # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
        'quantizable_layer_type': ['Conv2D', 'Linear'],
    }

def main(args):
    paddle.seed(12345)

    config = get_config(args.config, overrides=args.override, show=True)
    # assign the place
    use_gpu = config.get("use_gpu", True)
    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1
    config["use_data_parallel"] = use_data_parallel

    if config["use_data_parallel"]:
        paddle.distributed.init_parallel_env()

    net = program.create_model(config.ARCHITECTURE, config.classes_num)
    optimizer, lr_scheduler = program.create_optimizer(
        config, parameter_list=net.parameters())

    if config["use_data_parallel"]:
        net = paddle.DataParallel(net)

    # load model from checkpoint or pretrained model
    init_model(config, net, optimizer)
    quanter = QAT(config=quant_config)
    quanter.quantize(net)
    logger.info("QAT model summary:")
    paddle.summary(net, (1, 3, 224, 224))


    # save_model(net, optimizer, './test', 1)
    quanter.save_quantized_model(net, './quant_model', input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])

if __name__ == '__main__':
    args = parse_args()
    main(args)
