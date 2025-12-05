
from os.path import join, exists, basename
import argparse
from utils.prepar_for_fintune import *

def graph_to_tensor(atom_bond_graphs,bond_angle_graphs,dihedral_angle_graphs,conj_graph):
    return atom_bond_graphs.tensor(), bond_angle_graphs.tensor(), dihedral_angle_graphs.tensor(), conj_graph.tensor()

def train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt):
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
    for atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graph, valids, labels in data_gen:
        if len(labels) < args.batch_size * 0.5:
            continue

        atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graph = graph_to_tensor(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graph)
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graph)
        loss = criterion(preds, labels)
        loss = paddle.sum(loss * valids) / paddle.sum(valids)
        loss.backward()
        encoder_opt.step()
        head_opt.step()
        encoder_opt.clear_grad()
        head_opt.clear_grad()
        list_loss.append(loss.item())
    return np.mean(list_loss)



def evaluate(args, model, test_dataset, collate_fn):
    """
    Define the evaluate function
    """
    data_gen = test_dataset.get_data_loader(
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            shuffle=False,
            collate_fn=collate_fn)
    total_pred = []
    total_label = []
    total_valid = []
    model.eval()
    for atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs, valids, labels in data_gen:
        atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs = graph_to_tensor(
            atom_bond_graphs,bond_angle_graphs,angle_graphs,conj_graphs)
        labels = paddle.to_tensor(labels, 'float32')
        valids = paddle.to_tensor(valids, 'float32')
        preds = model(atom_bond_graphs, bond_angle_graphs, angle_graphs, conj_graphs)
        total_pred.append(preds.numpy())
        total_valid.append(valids.numpy())
        total_label.append(labels.numpy())
    total_pred = np.concatenate(total_pred, 0)
    total_label = np.concatenate(total_label, 0)
    total_valid = np.concatenate(total_valid, 0)
    return calc_rocauc_score(total_label, total_pred, total_valid)



def main(args):

    # initialize compound encoder and downstream model, get task names
    encoder_config, encoder, model, task_type, task_names = cla_initialize(args)
    # initialize loss and optimizer
    criterion, encoder_opt, head_opt = init_optimizer(args, encoder, model)
    dataset = load_data(args, task_names)
    # split dataset into train/valid/test data
    train_dataset, valid_dataset, test_dataset = cla_split_dataset(args, dataset)
    list(dataset).clear()
    # initialize collate_fn
    collate_fn = init_collate_fn(encoder_config, task_type)
    list_val_auc, list_test_auc = [], []
    for epoch_id in range(args.max_epoch):
        print("\nepoch:%s" % epoch_id)
        train_loss = train(args, model, train_dataset, collate_fn, criterion, encoder_opt, head_opt)
        list(train_dataset).clear()
        val_auc = evaluate(args, model, valid_dataset, collate_fn)
        list(valid_dataset).clear()
        test_auc = evaluate(args, model, test_dataset, collate_fn)
        list(test_dataset).clear()
        list_val_auc.append(val_auc)
        list_test_auc.append(test_auc)
        test_auc_by_eval = list_test_auc[np.argmax(list_val_auc)]
        print("epoch:%s train loss:%s" % (epoch_id, train_loss))
        print("epoch:%s valid auc:%s" % (epoch_id, val_auc))
        print("epoch:%s test auc:%s" % (epoch_id, test_auc))
        print("epoch:%s test auc_by_eval:%s" % (epoch_id, test_auc_by_eval))
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
        'exp_id': args.exp_id,
    }
    offset = 0  # 20
    best_epoch_id = np.argmax(list_val_auc[offset:]) + offset
    for metric, value in [
            ('test_auc', list_test_auc[best_epoch_id]),
            ('max_valid_auc', np.max(list_val_auc)),
            ('max_test_auc', np.max(list_test_auc))]:
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
    parser.add_argument("--dataset_name", default='bace', choices=['bace', 'bbbp', 'hiv', 'clintox', 'muv', 'sider', 'tox21', 'toxcast', 'pcba'])
    parser.add_argument("--split_type", default='scaffold', choices=['random', 'scaffold',  'index'])
    parser.add_argument("--subgraph_archi", type=str, default='ab')
    parser.add_argument("--encoder_config", type=str, default='model_configs/gnn_settings.json')
    parser.add_argument("--model_config", type=str, default='model_configs/down_mlp3.json')
    parser.add_argument("--init_model", type=str, default='pretrain_models/ene/epoch23.pdparams')
    parser.add_argument("--model_dir", type=str, default='downstream_models')
    parser.add_argument("--encoder_lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--head_lr", type=float, default=0.001)  # 0.001
    parser.add_argument("--dropout_rate", type=float, default=0.2)  # 0.2
    parser.add_argument("--exp_id", type=int, help='used for identification only')
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    main(args)
