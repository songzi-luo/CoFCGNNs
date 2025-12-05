
from os.path import join, exists, basename
import argparse
from src.util import calc_rmse, calc_mae
from utils.prepar_for_fintune import *

def graph_to_tensor(atom_bond_graphs,bond_angle_graphs,dihedral_angle_graphs,conj_graph):
    return atom_bond_graphs.tensor(), bond_angle_graphs.tensor(), dihedral_angle_graphs.tensor(), conj_graph.tensor()

def get_norm_label(labels,label_mean,label_std):
    norm_labels = (labels - label_mean) / (label_std + 1e-5)
    return paddle.to_tensor(norm_labels, 'float32')

def train(args, model, label_mean, label_std, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
    """
    Define the train function
    """
    data_gen = train_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=True,
            collate_fn=collate_fn)
    list_loss = []
    model.train()
    for atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue
        atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs = graph_to_tensor(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs)
        norm_labels = get_norm_label(labels=labels, label_mean=label_mean, label_std=label_std)
        preds = model(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs)
        loss = criterion(preds, norm_labels)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.item())
    return np.mean(list_loss)


def evaluate(args, model, label_mean, label_std, test_dataset, collate_fn, metric):
    """
    Define the evaluate function
    """
    data_all = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_pred = []
    total_label = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs, labels in data_all:
        atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs = graph_to_tensor(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs)
        labels = paddle.to_tensor(labels, 'float32')
        norm_preds = model(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs)
        preds = norm_preds.numpy() * label_std + label_mean
        total_pred.append(preds)
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    if metric == 'rmse':
        return calc_rmse(total_label, total_pred)
    else:
        return calc_mae(total_label, total_pred)



def main(args):
    """
    Call the configuration function of the model, build the model and load data, then start training.
    """

    # initialize compound encoder and downstream model
    encoder_config, compound_encoder, model_config, model, task_type, task_names, metric, label_mean, label_std = regr_initialize(args)
    # initialize loss and optimizer
    criterion, encoder_opt, head_opt = init_optimizer(args, compound_encoder, model, metric)

    # load raw data and tranform them
    dataset = load_data(args, task_names)

    # split dataset into train/valid/test data
    train_dataset, valid_dataset, test_dataset = regr_split_dataset(args, dataset)

    # initialize collate_fn
    collate_fn = init_collate_fn(encoder_config, task_type)

    ### start train
    list_val_metric, list_test_metric = [], []
    for epoch_id in range(args.max_epoch):
        print("\nepoch:%s" % epoch_id)
        train_loss = train(args, model, label_mean, label_std, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
        val_metric = evaluate(args, model, label_mean, label_std, valid_dataset, collate_fn, metric)
        test_metric = evaluate(args, model, label_mean, label_std, test_dataset, collate_fn, metric)
        list_val_metric.append(val_metric)
        list_test_metric.append(test_metric)
        test_metric_by_eval = list_test_metric[np.argmin(list_val_metric)]
        print("epoch:%s train/loss:%s" % (epoch_id, train_loss))
        print("epoch:%s val/%s:%s" % (epoch_id, metric, val_metric))
        print("epoch:%s test/%s:%s" % (epoch_id, metric, test_metric))
        print("epoch:%s test/%s_by_eval:%s" % (epoch_id, metric, test_metric_by_eval))
        paddle.save(model.state_dict(), '/%s_elr%s_hlr%s_dr%s/epoch%d/model.pdparams'
                    % (args.dataset_name, args.encoder_lr, args.head_lr, args.dropout_rate, epoch_id))
    outs = {
        'model_config': basename(args.model_config).replace('.json', ''),
        'metric': '',
        'dataset': args.dataset_name, 
        'split_type': args.split_type, 
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'encoder_lr': args.encoder_lr,
        'head_lr': args.head_lr,
    }
    best_epoch_id = np.argmin(list_val_metric)
    for metric, value in [
            ('test_%s' % metric, list_test_metric[best_epoch_id]),
            ('max_valid_%s' % metric, np.min(list_val_metric)),
            ('max_test_%s' % metric, np.min(list_test_metric))]:
        outs['metric'] = metric
        print('\t'.join(['FINAL'] + ["%s:%s" % (k, outs[k]) for k in outs] + [str(value)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=['train', 'data'], default='train')
    parser.add_argument("--batch_size", type=int, default=32)  # 32
    parser.add_argument("--num_workers", type=int, default=1)  # 2
    parser.add_argument("--max_epoch", type=int, default=100)  # 100
    parser.add_argument("--data_path", type=str, default='downstream_datasets')
    parser.add_argument("--cached_data_path", type=str, default='cached_data')
    parser.add_argument("--dataset_name", default='esol', choices=['esol', 'freesolv', 'lipophilicity','qm8','esol','qm7','qm9'])
    parser.add_argument("--split_type", default='scaffold', choices=['random', 'scaffold',  'index'])
    parser.add_argument("--encoder_config", type=str, default='model_configs/gnn_settings.json')
    parser.add_argument("--model_config", type=str, default='model_configs/down_mlp3.json')
    parser.add_argument("--init_model", type=str, default='pretrain_models/best.pdparams')  # GEM_pretrain_models/zinc_lr0001_bs512/epoch45.pdparams
    parser.add_argument("--model_dir", type=str, default='downstream_models')
    parser.add_argument("--encoder_lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--head_lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--dropout_rate", type=float, default=0.2)  # 0.2
    args = parser.parse_args()

    parser = argparse.ArgumentParser()
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    main(args)




