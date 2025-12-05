

import os
from os.path import join, exists, basename
import sys
import argparse
import time
import numpy as np
from glob import glob
import logging

import paddle
import paddle.distributed as dist

from datasets.Load_dataset import LoadDataset
from utils import load_json_config
from featurizers.feature_abstracter import PredCollateFn
from model_zoo.model import CoFCModel, spatial_head,energy_head

def train(args, model, optimizer, data_gen):
    """tbd"""
    model.train()
    print('in train')
    steps = get_steps_per_epoch(args)
    step = 0
    list_loss = []
    for graph_dict, label_dict in data_gen:
        print('rank:%s step:%s' % (dist.get_rank(), step))
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in label_dict:
            label_dict[k] = paddle.to_tensor(label_dict[k])
        # print(label_dict)
        train_loss = model(graph_dict, label_dict)
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        list_loss.append(float(train_loss))
        step += 1
        if step > steps:
            print("jumpping out")
            break
    return np.mean(list_loss)


@paddle.no_grad()
def evaluate(args, model, test_dataset, collate_fn):

    model.eval()
    data_gen = test_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn)
    dict_loss = {'loss': []}
    for graph_dict, label_dict in data_gen:
        for k in graph_dict:
            graph_dict[k] = graph_dict[k].tensor()
        for k in label_dict:
            label_dict[k] = paddle.to_tensor(label_dict[k])
        loss, sub_losses = model(graph_dict, label_dict, return_subloss=True)

        for name in sub_losses:
            if not name in dict_loss:
                dict_loss[name] = []

            v_np = float(sub_losses[name])
            dict_loss[name].append(v_np)
        dict_loss['loss'] = float(loss)
    dict_loss = {name: np.mean(dict_loss[name]) for name in dict_loss}
    return dict_loss


def get_steps_per_epoch(args):
    """tbd"""
    # add as argument
    steps_per_epoch = int(int(4000000 * (1 - args.test_ratio)) / args.batch_size)
    if args.distributed:
        steps_per_epoch = int(steps_per_epoch / dist.get_world_size())
    return steps_per_epoch



def main(args):
    """tbd"""
    print("support cuda:", paddle.is_compiled_with_cuda())
    print("num gpus:", paddle.device.cuda.device_count())
    if paddle.device.cuda.device_count() > 0:
        print("present device:", paddle.device.get_device())

    GNN_config = load_json_config(args.encoder_config)
    MLP_head_config = load_json_config(args.model_config)

    # get dataset
    dataset = LoadDataset(npz_data_path=args.data_path)
    dataset = dataset[dist.get_rank()::dist.get_world_size()]
    print('Total size:%s' % (len(dataset)))
    test_index = int(len(dataset) * (1 - args.test_ratio))
    train_dataset = dataset[:test_index]
    test_dataset = dataset[test_index:]
    print("Train/Test num: %s/%s" % (len(train_dataset), len(test_dataset)))

    collate_fn = PredCollateFn(MLP_head_config=MLP_head_config, encoder_config=GNN_config)
    train_data_gen = train_dataset.get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=collate_fn)
    print('graph feature over')
    print('batchsize =',args.batch_size)

    GNN_config = load_json_config(args.encoder_config)
    MLP_head_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        GNN_config['dropout_rate'] = args.dropout_rate
        MLP_head_config['dropout_rate'] = args.dropout_rate

    compound_encoder = CoFCModel(GNN_config)
    model = energy_head(MLP_head_config, compound_encoder)
    opt = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())


    list_test_loss = []
    ###
    def spatial_block_freezing(model, epoch, total_epochs):
        model.freeze_conj_only()
        phase = "spatial task"
        status = model.get_freezing_status()
        print(f"Epoch {epoch}/{total_epochs}: {phase} | freeze status: {status['geo_pred_config']}")
        return model

    ###
    for epoch_id in range(args.max_epoch):
        # s = time.time()
        print('activate')
        freeze_model = spatial_block_freezing(model=model, epoch=epoch_id, total_epochs=args.max_epoch)
        train_loss = train(args, freeze_model, opt, train_data_gen)
        test_loss = evaluate(args, freeze_model, test_dataset, collate_fn)
        if not args.distributed or dist.get_rank() == 0:
            paddle.save(compound_encoder.state_dict(),
                        '%s/epoch%d.pdparams' % (args.model_dir_ene, epoch_id))
            list_test_loss.append(test_loss['loss'])
            print("epoch:%d train/loss:%s" % (epoch_id, train_loss))
            print("epoch:%d test/loss:%s" % (epoch_id, test_loss))
            # print("Time used:%ss" % (time.time() - s))

    if not args.distributed or dist.get_rank() == 0:
        print('Best epoch id:%s' % np.argmin(list_test_loss))





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--DEBUG", action='store_true', default=False)
    parser.add_argument("--distributed", action='store_true', default=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--encoder_config", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--model_dir_spa", type=str,default='./pretrain_models/spa')
    parser.add_argument("--model_dir_ene", type=str,default='./pretrain_models/ene')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    args = parser.parse_args()
    if args.distributed:
        dist.init_parallel_env()
    main(args)