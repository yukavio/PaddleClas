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

    train_dataloader = Reader(config, 'train', places=place)()

    if config.validate:
        valid_dataloader = Reader(config, 'valid', places=place)()

    last_epoch_id = config.get("last_epoch", -1)
    best_top1_acc = 0.0  # best top1 acc record
    best_top1_epoch = last_epoch_id

    target_rate_dict = []
    drop_rate_dict = []
    cur_drop_rate = 0.5
    cur_target_rate = 0
    step = int(config.epochs / 2 / (args.target_rate+1))
    rise_point = list(range(0, int(config.epochs/2), step))
    for i in range(0, config.epochs):
        if i >= config.epochs/2:
            cur_drop_rate += 0.5 / (config.epochs/2)      
        if i in rise_point[1:]:
            cur_target_rate += 1
        drop_rate_dict.append(cur_drop_rate)   
        target_rate_dict.append(cur_target_rate)
        

    for epoch_id in range(last_epoch_id + 1, config.epochs):
        net.train()
        Real_sparse = []
        for n, m in net.named_sublayers():
            if 'GroupSparse' in n:
                m.set_target_rate(target_rate_dict[epoch_id])
                m.set_drop_rate(drop_rate_dict[epoch_id])
                Real_sparse.append(m.get_real_sparsity())
                Infer_sparse = m.get_infer_sparsity()
        logger.info("Target Rate:{}  Drop Rate {}: Infer Sparse: {:.5} Real Sparse:".format(
            target_rate_dict[epoch_id], drop_rate_dict[epoch_id], Infer_sparse
        ))
        logger.info(Real_sparse)
        # 1. train with train dataset
        program.run(train_dataloader, config, net, optimizer, lr_scheduler,
                    epoch_id, 'train')

        # 2. validate with validate dataset
        if config.validate and epoch_id % config.valid_interval == 0:
            net.eval()
            with paddle.no_grad():
                top1_acc = program.run(valid_dataloader, config, net, None,
                                       None, epoch_id, 'valid')
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top1_epoch = epoch_id
                if epoch_id % config.save_interval == 0:
                    model_path = os.path.join(config.model_save_dir,
                                              config.ARCHITECTURE["name"])
                    save_model(net, optimizer, model_path, "best_model")
            message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                best_top1_acc, best_top1_epoch)
            logger.info("{:s}".format(logger.coloring(message, "RED")))

        # 3. save the persistable model
        if epoch_id % config.save_interval == 0:
            model_path = os.path.join(config.model_save_dir,
                                      config.ARCHITECTURE["name"])
            save_model(net, optimizer, model_path, epoch_id)


if __name__ == '__main__':
    args = parse_args()
    main(args)
