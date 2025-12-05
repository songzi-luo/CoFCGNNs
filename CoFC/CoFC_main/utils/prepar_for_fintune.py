import numpy as np
import paddle
import paddle.nn as nn
from model_zoo.model import CoFCModel
from utils import load_json_config
from datasets.Load_dataset import LoadDataset
from src.model import encoder_with_head
from src.featurizer import DownstreamTransformFn, DownstreamCollateFn
from src.util import calc_rocauc_score, exempt_parameters,ScaffoldSplitter
from datasets import downstream_datasets

def get_metric(dataset_name):
    """tbd"""
    if dataset_name in ['esol', 'freesolv', 'lipophilicity']:
        return 'rmse'
    elif dataset_name in ['qm7', 'qm8', 'qm9']:
        return 'mae'
    else:
        raise ValueError(dataset_name)

def cla_initialize(args):
    encoder_config = load_json_config(args.encoder_config)
    if not args.dropout_rate is None:
        encoder_config['dropout_rate'] = args.dropout_rate
    print('encoder_config:')
    print(encoder_config)
    task_type = 'class'
    head_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        head_config['dropout_rate'] = args.dropout_rate
    task_names = downstream_datasets.get_downstream_task_names(args.dataset_name,args.data_path + "/" + args.dataset_name)
    head_config['task_type'] = task_type
    head_config['num_tasks'] = len(task_names)
    print('head_config:')
    print(head_config)
    encoder = CoFCModel(encoder_config)
    if not args.init_model is None and not args.init_model == "":
        encoder.set_state_dict(paddle.load(args.init_model))
        print('Load state_dict from %s' % args.init_model)
    model = encoder_with_head(head_config, encoder)
    return encoder_config, encoder, model, task_type, task_names

def regr_initialize(args):

    encoder_config = load_json_config(args.encoder_config)
    if not args.dropout_rate is None:
        encoder_config['dropout_rate'] = args.dropout_rate
    encoder_config['subgraph_archi'] = args.subgraph_archi
    print('===encoder_config===')
    print(encoder_config)

    task_type = 'regr'
    metric = get_metric(args.dataset_name)
    task_names = downstream_datasets.get_downstream_task_names(args.dataset_name,args.data_path + "/" + args.dataset_name)
    dataset_stat = downstream_datasets.get_downstream_data_stat(args.dataset_name, args.data_path + "/" + args.dataset_name, task_names)
    label_mean = np.reshape(dataset_stat['mean'], [1, -1])
    label_std = np.reshape(dataset_stat['std'], [1, -1])

    model_config = load_json_config(args.model_config)
    if not args.dropout_rate is None:
        model_config['dropout_rate'] = args.dropout_rate
    model_config['task_type'] = task_type
    model_config['num_tasks'] = len(task_names)
    print('===model_config===')
    print(model_config)

    ### build model
    compound_encoder = CoFCModel(encoder_config)
    model = encoder_with_head(model_config, compound_encoder)
    return encoder_config, compound_encoder, model_config, model, task_type, task_names, metric, label_mean, label_std
"""
function: init_loss_optimizer()
return: criterion, encoder_opt, head_opt
"""
def init_optimizer(args, encoder, model,metric = False):
    """
    Define loss, Adam optimizer
    """
    if metric in ['rmse' , 'mae']:
        criterion = nn.L1Loss()
    else:
        criterion = nn.BCELoss(reduction='none')
    encoder_params = encoder.parameters()
    head_params = exempt_parameters(model.parameters(), encoder_params)
    encoder_opt = paddle.optimizer.Adam(args.encoder_lr, parameters=encoder_params)
    head_opt = paddle.optimizer.Adam(args.head_lr, parameters=head_params)
    print('Total param num: %s' % (len(model.parameters())))
    print('Encoder param num: %s' % (len(encoder_params)))
    print('Head param num: %s' % (len(head_params)))
    return criterion, encoder_opt, head_opt


"""
function: load_data()
return: dataset
"""
def load_data(args, task_names):
    """
    featurizer:
        Gen features according to the raw data and return the graph data.
        Collate features about the graph data and return the feed dictionary.
    """
    cached_data_path_datesetname = args.cached_data_path + "/" + args.dataset_name
    if args.task == 'data':
        print('Preprocessing data...')
        dataset = downstream_datasets.load_downstream_dataset(args.dataset_name,
                                                              args.data_path + "/" + args.dataset_name, task_names)

        dataset.transform(DownstreamTransformFn(), num_workers=5)
        print('Dataset len: %s' % len(dataset))
        print("Saving data...")
        print("Finished")
        dataset.save_data(cached_data_path_datesetname)
        exit(0)
    else:
        if cached_data_path_datesetname is None or cached_data_path_datesetname == "":
            print('Processing data...')
            dataset = downstream_datasets.load_downstream_dataset(args.dataset_name, args.data_path + "/" + args.dataset_name, task_names)
            dataset.transform(DownstreamTransformFn(), num_workers=5)
            print('Dataset len: %s' % len(dataset))
        else:
            print('Read preprocessing data...')
            dataset = LoadDataset(npz_data_path=cached_data_path_datesetname)
            print('Dataset len: %s' % len(dataset))
    return dataset


"""
function: get_pos_neg_ratio()
return: pos vs neg ratio
"""
def get_pos_neg_ratio(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.mean(labels == 1), np.mean(labels == -1)

"""
function: split_dataset()
return: train_dataset, valid_dataset, test_dataset
"""
def cla_split_dataset(args, dataset):
    """
    Split dataset by ScaffoldSplitter"
    """
    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
        len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train pos/neg ratio %s/%s' % get_pos_neg_ratio(train_dataset))
    print('Valid pos/neg ratio %s/%s' % get_pos_neg_ratio(valid_dataset))
    print('Test pos/neg ratio %s/%s' % get_pos_neg_ratio(test_dataset))
    return train_dataset, valid_dataset, test_dataset

def get_label_stat(dataset):
    """tbd"""
    labels = np.array([data['label'] for data in dataset])
    return np.min(labels), np.max(labels), np.mean(labels)

def regr_split_dataset(args, dataset):
    splitter = ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    print("Train/Valid/Test num: %s/%s/%s" % (
        len(train_dataset), len(valid_dataset), len(test_dataset)))
    print('Train label: min/max/mean %s/%s/%s' % get_label_stat(train_dataset))
    print('Valid label: min/max/mean %s/%s/%s' % get_label_stat(valid_dataset))
    print('Test label: min/max/mean %s/%s/%s' % get_label_stat(test_dataset))
    return train_dataset, valid_dataset, test_dataset


def init_collate_fn(encoder_config, task_type):

    collate_fn = DownstreamCollateFn(
            atom_names=encoder_config['atom_names'],
            # atomic_num, formal_charge, degree, chiral_tag, total_numHs, is_aromatic, hybridization
            bond_names=encoder_config['bond_names'],  # bond_dir, bond_type, is_in_ring
            bond_float_names=encoder_config['bond_float_names'],  # bond_length
            bond_angle_float_names=encoder_config['bond_angle_float_names'],  # bond_angle
            plane_names=encoder_config['plane_names'],  # plane_in_ring
            plane_float_names=encoder_config['plane_float_names'],  # plane_mass
            dihedral_angle_float_names=encoder_config['dihedral_angle_float_names'],  # DihedralAngleGraph_angles
            task_type=task_type)
    return collate_fn

def graph_to_tensor(atom_bond_graphs,bond_angle_graphs,dihedral_angle_graphs,conj_graph):
    return atom_bond_graphs.tensor(), bond_angle_graphs.tensor(), dihedral_angle_graphs.tensor(), conj_graph.tensor()