

import numpy as np
import os
import random
import json

from pgl.utils.data import Dataloader


def mp_pool_map(list_input, func, num_workers):
    """list_output = [func(input) for input in list_input]"""
    class _CollateFn(object):
        def __init__(self, func):
            self.func = func
        def __call__(self, data_list):
            new_data_list = []
            for data in data_list:
                index, input = data
                new_data_list.append((index, self.func(input)))
            return new_data_list

    # add index
    list_new_input = [(index, x) for index, x in enumerate(list_input)]
    data_gen = Dataloader(list_new_input, 
            batch_size=8, 
            num_workers=num_workers, 
            shuffle=False,
            collate_fn=_CollateFn(func))  

    list_output = []
    for sub_outputs in data_gen:
        list_output += sub_outputs
    list_output = sorted(list_output, key=lambda x: x[0])
    # remove index
    list_output = [x[1] for x in list_output]
    # print('list_output', list_output[0])
    return list_output


def load_json_config(path):
    """tbd"""
    return json.load(open(path, 'r'))