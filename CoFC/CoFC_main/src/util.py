#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
utils
"""

from __future__ import print_function
import sys
import os
from os.path import exists, dirname
import numpy as np
import pickle
import json
import time
import six
if six.PY3:
    import _thread as thread
    from queue import Queue
from collections import OrderedDict
from datetime import datetime
from sklearn.metrics import roc_auc_score
import multiprocessing
import paddle.distributed as dist
from glob import glob
from utils.splitters import *
from datasets.Load_dataset import LoadDataset


def calc_rocauc_score(labels, preds, valid):
    """compute ROC-AUC and averaged across tasks"""
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    rocauc_list = []
    for i in range(labels.shape[1]):
        c_valid = valid[:, i].astype("bool")
        c_label, c_pred = labels[c_valid, i], preds[c_valid, i]
        #AUC is only defined when there is at least one positive data.
        if len(np.unique(c_label)) == 2:
            rocauc_list.append(roc_auc_score(c_label, c_pred))

    print('Valid ratio: %s' % (np.mean(valid)))
    print('Task evaluated: %s/%s' % (len(rocauc_list), labels.shape[1]))
    if len(rocauc_list) == 0:
        raise RuntimeError("No positively labeled data available. Cannot compute ROC-AUC.")

    return sum(rocauc_list)/len(rocauc_list)


def calc_rmse(labels, preds):
    """tbd"""
    return np.sqrt(np.mean((preds - labels) ** 2))


def calc_mae(labels, preds):
    """tbd"""
    return np.mean(np.abs(preds - labels))


def exempt_parameters(src_list, ref_list):
    """Remove element from src_list that is in ref_list"""
    res = []
    for x in src_list:
        flag = True
        for y in ref_list:
            if x is y:
                flag = False
                break
        if flag:
            res.append(x)
    return res


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def avg_split_list(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def load_pkls_to_list(args):
    fid, pkl_path = args
    if (pkl_path.endswith(".pkl")):
        pkl = open(pkl_path, "rb")
        data = pickle.load(pkl)
        if fid % 10 == 0:
            print("  ", fid, end=", ")
    return data


def get_pickle_files_list(path):
    # traversal directory
    files_list = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".pkl"):
                files_list.append(os.path.join(root, name))
    files_list.sort()
    return files_list


"""
Load data, build dataset list with InMemoryDataset, each line is the smile of a molecular
"""
def load_smiles_to_dataset(data_path):
    """tbd"""
    files = sorted(glob('%s/*' % data_path))
    print("files:", files)
    data_list = []
    for file in files:
        with open(file, 'r') as f:
            tmp_data_list = [line.strip() for line in f.readlines()]
        data_list.extend(tmp_data_list)
    dataset = LoadDataset(data_list=data_list)
    return dataset


def get_steps_per_epoch(args):
    """tbd"""
    # add as argument
    if args.dataset == 'zinc':
        train_num = int(20000000 * (1 - args.test_ratio))
    else:
        raise ValueError(args.dataset)
    # if args.DEBUG:
    #     train_num = 100
    steps_per_epoch = int(train_num / args.batch_size)
    if args.distributed:
        steps_per_epoch = int(steps_per_epoch / dist.get_world_size())
    return steps_per_epoch

