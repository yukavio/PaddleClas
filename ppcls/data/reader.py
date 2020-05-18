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

import numpy as np
import imghdr
import os
import signal

from paddle.fluid.io import multiprocess_reader

from . import imaug
from .imaug import transform
from ppcls.utils import logger

trainers_num = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
trainer_id = int(os.environ.get("PADDLE_TRAINER_ID", 0))


class ModeException(Exception):
    """
    ModeException
    """

    def __init__(self, message='', mode=''):
        message += "\nOnly the following 3 modes are supported: " \
            "train, valid, test. Given mode is {}".format(mode)
        super(ModeException, self).__init__(message)


class SampleNumException(Exception):
    """
    SampleNumException
    """

    def __init__(self, message='', sample_num=0, batch_size=1):
        message += "\nError: The number of the whole data ({}) " \
            "is smaller than the batch_size ({}), and drop_last " \
            "is turnning on, so nothing  will feed in program, " \
            "Terminated now. Please reset batch_size to a smaller " \
            "number or feed more data!".format(sample_num, batch_size)
        super(SampleNumException, self).__init__(message)


class ShuffleSeedException(Exception):
    """
    ShuffleSeedException
    """

    def __init__(self, message=''):
        message += "\nIf trainers_num > 1, the shuffle_seed must be set, " \
            "because the order of batch data generated by reader " \
            "must be the same in the respective processes."
        super(ShuffleSeedException, self).__init__(message)


def check_params(params):
    """
    check params to avoid unexpect errors

    Args:
        params(dict):
    """
    if 'shuffle_seed' not in params:
        params['shuffle_seed'] = None

    if trainers_num > 1 and params['shuffle_seed'] is None:
        raise ShuffleSeedException()

    data_dir = params.get('data_dir', '')
    assert os.path.isdir(data_dir), \
        "{} doesn't exist, please check datadir path".format(data_dir)

    if params['mode'] != 'test':
        file_list = params.get('file_list', '')
        assert os.path.isfile(file_list), \
            "{} doesn't exist, please check file list path".format(file_list)


def create_file_list(params):
    """
    if mode is test, create the file list

    Args:
        params(dict):
    """
    data_dir = params.get('data_dir', '')
    params['file_list'] = ".tmp.txt"
    imgtype_list = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff'}
    with open(params['file_list'], "w") as fout:
        tmp_file_list = os.listdir(data_dir)
        for file_name in tmp_file_list:
            file_path = os.path.join(data_dir, file_name)
            if imghdr.what(file_path) not in imgtype_list:
                continue
            fout.write(file_name + " 0" + "\n")


def shuffle_lines(full_lines, seed=None):
    """
    random shuffle lines

    Args:
        full_lines(list):
        seed(int): random seed
    """
    if seed is not None:
        np.random.RandomState(seed).shuffle(full_lines)
    else:
        np.random.shuffle(full_lines)

    return full_lines


def get_file_list(params):
    """
    read label list from file and shuffle the list

    Args:
        params(dict):
    """
    if params['mode'] == 'test':
        create_file_list(params)

    with open(params['file_list']) as flist:
        full_lines = [line.strip() for line in flist]

    full_lines = shuffle_lines(full_lines, params["shuffle_seed"])

    # use only partial data for each trainer in distributed training
    if params['mode'] == 'train':
        img_per_trainer = len(full_lines) // trainers_num
        full_lines = full_lines[trainer_id::trainers_num][:img_per_trainer]

    return full_lines


def create_operators(params):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(imaug, op_name)(**param)
        ops.append(op)

    return ops


def partial_reader(params, full_lines, part_id=0, part_num=1):
    """
    create a reader with partial data

    Args:
        params(dict):
        full_lines: label list
        part_id(int): part index of the current partial data
        part_num(int): part num of the dataset
    """
    assert part_id < part_num, ("part_num: {} should be larger "
                                "than part_id: {}".format(part_num, part_id))

    full_lines = full_lines[part_id::part_num]

    batch_size = int(params['batch_size']) // trainers_num
    if params['mode'] != "test" and len(full_lines) < batch_size:
        raise SampleNumException('', len(full_lines), batch_size)

    def reader():
        ops = create_operators(params['transforms'])
        for line in full_lines:
            img_path, label = line.split()
            img_path = os.path.join(params['data_dir'], img_path)
            with open(img_path, 'rb') as f:
                img = f.read()
            yield (transform(img, ops), int(label))

    return reader


def mp_reader(params):
    """
    multiprocess reader

    Args:
        params(dict):
    """
    check_params(params)

    full_lines = get_file_list(params)
    if params["mode"] == "train":
        full_lines = shuffle_lines(full_lines, seed=None)

    part_num = 1 if 'num_workers' not in params else params['num_workers']

    readers = []
    for part_id in range(part_num):
        readers.append(partial_reader(params, full_lines, part_id, part_num))

    return multiprocess_reader(readers, use_pipe=False)


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    logger.info("main proc {} exit, kill process group "
                "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


class Reader:
    """
    Create a reader for trainning/validate/test

    Args:
        config(dict): arguments
        mode(str): train or val or test
        seed(int): random seed used to generate same sequence in each trainer

    Returns:
        the specific reader
    """

    def __init__(self, config, mode='train', seed=None):
        try:
            self.params = config[mode.upper()]
        except KeyError:
            raise ModeException(mode=mode)

        use_mix = config.get('use_mix')
        self.params['mode'] = mode
        if seed is not None:
            self.params['shuffle_seed'] = seed
        self.batch_ops = []
        if use_mix and mode == "train":
            self.batch_ops = create_operators(self.params['mix'])

    def __call__(self):
        batch_size = int(self.params['batch_size']) // trainers_num

        def wrapper():
            reader = mp_reader(self.params)
            batch = []
            for idx, sample in enumerate(reader()):
                img, label = sample
                batch.append((img, label))
                if (idx + 1) % batch_size == 0:
                    batch = transform(batch, self.batch_ops)
                    yield batch
                    batch = []

        return wrapper


signal.signal(signal.SIGINT, term_mp)
signal.signal(signal.SIGTERM, term_mp)
