
"""
This is an implementation of GeoGNN:
"""
import numpy as np

import paddle
import paddle.nn as nn
import pgl
from pgl.nn import GraphPool

from networks.gnn_block import GIN
from networks.compound_encoder import AtomEmbedding, BondEmbedding, \
        BondFloatRBF, BondAngleFloatRBF, PlaneEmbedding, PlaneFloatRBF, DihedralAngleFloatRBF,conj_edge_RBF
from utils.compound_tools import CompoundKit
from networks.gnn_block import MeanPool, GraphNorm
from networks.basic_block import MLP


class GNNBlock(nn.Layer):
    """
    GNN Block
    """
    def __init__(self, embed_dim, dropout_rate, last_act):
        super(GNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GIN(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.graph_norm = GraphNorm()
        if last_act:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, graph, node_hidden, edge_hidden):
        """tbd"""
        out = self.gnn(graph, node_hidden, edge_hidden)
        out = self.norm(out)
        out = self.graph_norm(graph, out)
        if self.last_act:
            out = self.act(out)
        out = self.dropout(out)
        out = out + node_hidden
        return out


class CoFCModel(nn.Layer):

    def __init__(self, model_config={}):
        super(CoFCModel, self).__init__()
        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')
        self.plane_names = model_config["plane_names"]
        self.plane_float_names = model_config["plane_float_names"]
        self.dihedral_angle_float_names = model_config["dihedral_angle_float_names"]
        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.bond_angle_float_names = model_config["bond_angle_float_names"]
        self.conj_atom_name = model_config["conj_atom_names"]
        self.conj_edge_name = model_config["conj_edge_names"]

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)

        self.init_plane_embedding = PlaneEmbedding(self.plane_names, self.embed_dim)
        self.init_plane_float_rbf = PlaneFloatRBF(self.plane_float_names, self.embed_dim)


        self.bond_embedding_list = nn.LayerList()
        self.bond_float_rbf_list = nn.LayerList()
        self.atom_bond_block_list = nn.LayerList()
        self.bond_angle_float_rbf_list = nn.LayerList()
        self.bond_angle_block_list = nn.LayerList()

        self.plane_embedding_list = nn.LayerList()
        self.plane_float_rbf_list = nn.LayerList()
        self.dihedral_angle_float_rbf_list = nn.LayerList()
        self.dihedral_angle_block_list = nn.LayerList()
        self.init_conj_edge_embedding = BondEmbedding(self.conj_edge_name, self.embed_dim)
        self.init_conj_node_embedding = AtomEmbedding(self.conj_atom_name, self.embed_dim)
        self.conj_node_block_list = nn.LayerList()
        self.conj_edge_block_list = nn.LayerList()
        self.conj_edge_rbf_list = nn.LayerList()
        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                    BondEmbedding(self.bond_names, self.embed_dim))
            self.bond_float_rbf_list.append(
                    BondFloatRBF(self.bond_float_names, self.embed_dim))
            self.atom_bond_block_list.append(
                GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim))
            self.bond_angle_block_list.append(
                GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.plane_embedding_list.append(
                PlaneEmbedding(self.plane_names, self.embed_dim))
            self.plane_float_rbf_list.append(
                PlaneFloatRBF(self.plane_float_names, self.embed_dim))
            self.dihedral_angle_float_rbf_list.append(
                    DihedralAngleFloatRBF(self.dihedral_angle_float_names, self.embed_dim))
            self.dihedral_angle_block_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))

            self.conj_edge_rbf_list.append(conj_edge_RBF(self.conj_edge_name, self.embed_dim))
            self.conj_node_block_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))
            self.conj_edge_block_list.append(
                    GNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1)))

        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = pgl.nn.GraphPool(pool_type=self.readout)
        self.freeze_config = {
            'conj_layers': False,
            'bond_angle_layers': False,
            'freeze_embedding': False
        }

        print('embed_dim:%s' % self.embed_dim)
        print('dropout_rate:%s' % self.dropout_rate)
        print(' layer_num:%s' % self.layer_num)
        print('readout:%s' % self.readout)
        print('atom_features:%s' % str(self.atom_names))
        print('bond_features:%s' % str(self.bond_names))
        print('bond_float_names:%s' % str(self.bond_float_names))
        print('bond_angle_float_names:%s' % str(self.bond_angle_float_names))
        print('plane_names:%s' % str(self.plane_names))
        print('plane_float_names:%s' % str(self.plane_float_names))
        print('dihedral_angle_float_names:%s' % str(self.dihedral_angle_float_names))


    def _apply_freezing_config(self):
        """freeze block by input"""

        if self.freeze_config['conj_layers']:
            self._freeze_conj_layers(self.freeze_config['freeze_embedding'])

        if self.freeze_config['bond_angle_layers']:
            self._freeze_bond_angle_layers(self.freeze_config['freeze_embedding'])

        self._print_freezing_status()

    def _freeze_conj_layers(self, freeze_embedding=True):
        """freeze conj block"""

        for param in self.init_conj_edge_embedding.parameters():
            param.requires_grad = False
        for param in self.init_conj_node_embedding.parameters():
            param.requires_grad = False


        for i in range(self.layer_num):
            for param in self.conj_node_block_list[i].parameters():
                param.requires_grad = False
            for param in self.conj_edge_block_list[i].parameters():
                param.requires_grad = False
            for param in self.conj_edge_rbf_list[i].parameters():
                param.requires_grad = False

    def _freeze_bond_angle_layers(self, freeze_embedding=True):
        """freeze spatial block"""

        if freeze_embedding:
            for param in self.init_bond_embedding.parameters():
                param.requires_grad = False
            for param in self.init_bond_float_rbf.parameters():
                param.requires_grad = False
            for param in self.init_plane_embedding.parameters():
                param.requires_grad = False
            for param in self.init_plane_float_rbf.parameters():
                param.requires_grad = False


        for i in range(self.layer_num):
            for param in self.bond_float_rbf_list[i].parameters():
                param.requires_grad = False
            for param in self.bond_angle_float_rbf_list[i].parameters():
                param.requires_grad = False
            for param in self.bond_angle_block_list[i].parameters():
                param.requires_grad = False

            for param in self.plane_embedding_list[i].parameters():
                param.requires_grad = False
            for param in self.plane_float_rbf_list[i].parameters():
                param.requires_grad = False
            for param in self.dihedral_angle_float_rbf_list[i].parameters():
                param.requires_grad = False
            for param in self.dihedral_angle_block_list[i].parameters():
                param.requires_grad = False

    def _print_freezing_status(self):
        """print freezing status"""
        status = []
        if self.freeze_config['conj_layers']:
            status.append("conj layer")
        if self.freeze_config['bond_angle_layers']:
            status.append("bond angle layer")

        if status:
            print(f"=====freeze setting: {', '.join(status)}=====")
        else:
            print("freeze setting: non freeze")



    def set_freezing(self, conj_layers=None, bond_angle_layers=None, freeze_embedding=None):
        """set freezing"""
        if conj_layers is not None:
            self.freeze_config['conj_layers'] = conj_layers
        if bond_angle_layers is not None:
            self.freeze_config['bond_angle_layers'] = bond_angle_layers
        if freeze_embedding is not None:
            self.freeze_config['freeze_embedding'] = freeze_embedding

        self._apply_freezing_config()

    def freeze_conj_only(self):
        """freeze conj only"""
        self.set_freezing(conj_layers=True, bond_angle_layers=False)

    def freeze_bond_angle_only(self):
        """freeze spatial only"""
        self.set_freezing(conj_layers=False, bond_angle_layers=True)

    def freeze_both(self):
        """freeze all"""
        self.set_freezing(conj_layers=True, bond_angle_layers=True)

    def unfreeze_all(self):
        """unfreezeing"""
        print('freeze all block')
        self.set_freezing(conj_layers=False, bond_angle_layers=False)

    def get_freezing_status(self):
        """print present freezing status"""
        conj_params = sum(p.numel() for p in self.conj_node_block_list[0].parameters())
        bond_angle_params = sum(p.numel() for p in self.bond_angle_block_list[0].parameters())

        trainable_params = sum(p.numel() for p in self.parameters() if p.stop_gradient)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.stop_gradient)

        return {
            'conj_frozen': self.freeze_config['conj_layers'],
            'bond_angle_frozen': self.freeze_config['bond_angle_layers'],
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'total_params': trainable_params + frozen_params,
            'freeze_ratio': frozen_params / (trainable_params + frozen_params)
        }


    @property
    def node_dim(self):
        """the out dim of node_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph, dihedral_angle_graph, conj_graph):
        """
        Build the network.
        """

        node_hidden = self.init_atom_embedding(atom_bond_graph.node_feat)
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_feat)
        edge_hidden = bond_embed + self.init_bond_float_rbf(atom_bond_graph.edge_feat)

        node_hidden_list = [node_hidden]
        edge_hidden_list = [edge_hidden]
        plane_hidden_list = []

        plane_embed = self.init_plane_embedding(dihedral_angle_graph.node_feat)
        plane_hidden = plane_embed + self.init_plane_float_rbf(dihedral_angle_graph.node_feat)
        plane_hidden_list.append(plane_hidden)

        conj_edge_hidden = self.init_conj_edge_embedding(conj_graph.edge_feat)
        conj_edge_hidden_list = [conj_edge_hidden]
        conj_node_hidden = self.init_conj_node_embedding(conj_graph.node_feat)
        conj_node_hidden_list = [conj_node_hidden]

        for layer_id in range(self.layer_num):
            node_hidden = self.atom_bond_block_list[layer_id](
                atom_bond_graph,
                node_hidden_list[layer_id],
                edge_hidden_list[layer_id])

            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_feat)
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_feat)
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_feat)
            cur_angle_hidden = cur_angle_hidden + plane_hidden_list[layer_id]
            edge_hidden = self.bond_angle_block_list[layer_id](
                bond_angle_graph,
                cur_edge_hidden,
                cur_angle_hidden)

            cur_plane_hidden = self.plane_embedding_list[layer_id](dihedral_angle_graph.node_feat)
            cur_plane_hidden = cur_plane_hidden + self.plane_float_rbf_list[layer_id](
                dihedral_angle_graph.node_feat)
            cur_dihedral_angle_hidden = self.dihedral_angle_float_rbf_list[layer_id](dihedral_angle_graph.edge_feat)
            plane_hidden = self.dihedral_angle_block_list[layer_id](
                dihedral_angle_graph,
                cur_plane_hidden,
                cur_dihedral_angle_hidden)

            conj_edge_hidden = self.conj_edge_rbf_list[layer_id](conj_graph.edge_feat)
            conj_node_hidden = self.conj_node_block_list[layer_id](
                conj_graph,
                conj_node_hidden_list[layer_id],
                conj_edge_hidden_list[layer_id])
            #node level merge
            node_hidden = conj_node_hidden + node_hidden
            conj_edge_hidden_list.append(conj_edge_hidden)
            conj_node_hidden_list.append(conj_node_hidden)

            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)
            plane_hidden_list.append(plane_hidden)

        node_repr = node_hidden_list[-1]
        edge_repr = edge_hidden_list[-1]
        plane_repr = []
        if len(plane_hidden_list) != 0: plane_repr = plane_hidden_list[-1]
        graph_repr = self.graph_pool(atom_bond_graph, node_repr)
        return node_repr, edge_repr, plane_repr, graph_repr

class energy_head(nn.Layer):
    def __init__(self, model_config, compound_encoder):
        super(energy_head, self).__init__()
        self.compound_encoder = compound_encoder
        self.hidden_size = model_config['hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.act = model_config['act']
        self.pretrain_tasks = model_config['pretrain_tasks']

        self.homo_loss = nn.SmoothL1Loss()
        self.lumo_loss = nn.SmoothL1Loss()
        self.dipole_loss = nn.SmoothL1Loss()
        self.mulliken_loss = nn.SmoothL1Loss()
        # homo,lumo,dip,muliken
        self.homo_Linear = nn.Linear(compound_encoder.embed_dim, 1)
        self.lumo_Linear = nn.Linear(compound_encoder.embed_dim, 1)
        self.dipole_Linear = nn.Linear(compound_encoder.embed_dim, 1)
        self.mulliken_mlp = MLP(2,
                                hidden_size=self.hidden_size,
                                act=self.act,
                                in_size=compound_encoder.embed_dim,
                                out_size=1,
                                dropout_rate=self.dropout_rate)

        #################
        self.freeze_config = {
            'conj_layers': False,
            'bond_angle_layers': False,
            'freeze_embedding': False
        }
    def freeze_conj_only(self):
        """freeze conj"""
        if hasattr(self.compound_encoder, 'freeze_conj_only'):
            self.compound_encoder.freeze_conj_only()
            self.freeze_config.update({'conj_layers': True, 'bond_angle_layers': False})
        else:
            print("warning: compound_encoder not support freeze_conj_only")

    def freeze_bond_angle_only(self):
        """freeze spatial"""
        if hasattr(self.compound_encoder, 'freeze_bond_angle_only'):
            self.compound_encoder.freeze_bond_angle_only()
            self.freeze_config.update({'conj_layers': False, 'bond_angle_layers': True})
        else:
            print("warning: compound_encoder not support freeze_bond_angle_only")

    def freeze_both(self):
        """freeze all"""
        if hasattr(self.compound_encoder, 'freeze_both'):
            self.compound_encoder.freeze_both()
            self.freeze_config.update({'conj_layers': True, 'bond_angle_layers': True})
        else:
            print("warning: compound_encoder not support freeze_both")

    def unfreeze_all(self):
        """unfreeze all"""
        if hasattr(self.compound_encoder, 'unfreeze_all'):
            self.compound_encoder.unfreeze_all()
            self.freeze_config.update({'conj_layers': False, 'bond_angle_layers': False})
        else:
            print("warning: compound_encoder not support unfreeze_all")

    def get_freezing_status(self):
        """gain freeing status"""
        if hasattr(self.compound_encoder, 'get_freezing_status'):
            status = self.compound_encoder.get_freezing_status()
            status['geo_pred_config'] = self.freeze_config.copy()
            return status
        else:
            return {
                'geo_pred_config': self.freeze_config.copy(),
                'message': 'compound_encoder 不支持状态查询'
            }
    def _get_homo_loss(self, feed_dict, graph_repr):
        masked_graph_repr = graph_repr
        pred = self.homo_Linear(masked_graph_repr)
        loss = self.homo_loss(pred, feed_dict['homo'])
        return loss

    def _get_lumo_loss(self, feed_dict, graph_repr):
        masked_graph_repr = graph_repr
        pred = self.lumo_Linear(masked_graph_repr)
        loss = self.lumo_loss(pred, feed_dict['lumo'])
        return loss

    def _get_dipole_loss(self, feed_dict, graph_repr):
        masked_graph_repr = graph_repr
        pred = self.dipole_Linear(masked_graph_repr)
        loss = self.dipole_loss(pred, feed_dict['dipole'])
        return loss

    def _get_mulliken_loss(self, feed_dict, node_repr, graph_dict):
        pred = self.mulliken_mlp(node_repr)
        loss = self.mulliken_loss(pred, feed_dict['mulliken_charges'])
        return loss

    def forward(self, graph_dict, label_dict, return_subloss=False):
        """
        Build the loss.
        """

        node_repr, edge_repr, plane_repr, graph_repr = self.compound_encoder.forward(
            graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'], graph_dict['dihedral_angle_graph'],
            graph_dict['conj_graph'])
        masked_node_repr, masked_edge_repr, masked_plane_repr, masked_graph_repr = self.compound_encoder.forward(
            graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'],
            graph_dict['masked_dihedral_angle_graph'], graph_dict['conj_graph'])

        sub_losses = {}
        sub_losses['homo_loss'] = self._get_homo_loss(label_dict, graph_repr)
        sub_losses['homo_loss'] += self._get_homo_loss(label_dict, masked_graph_repr)
        sub_losses['lumo_loss'] = self._get_lumo_loss(label_dict, graph_repr)
        sub_losses['lumo_loss'] += self._get_lumo_loss(label_dict, masked_graph_repr)
        sub_losses['dipole_loss'] = self._get_dipole_loss(label_dict, graph_repr)
        sub_losses['dipole_loss'] += self._get_dipole_loss(label_dict, masked_graph_repr)
        sub_losses['mulliken_loss'] = self._get_mulliken_loss(label_dict, node_repr, graph_dict)
        sub_losses['mulliken_loss'] += self._get_mulliken_loss(label_dict, masked_node_repr, graph_dict)

        loss = 0
        for name in sub_losses:
            loss += sub_losses[name]
        if return_subloss:
            return loss, sub_losses
        else:
            return loss







class spatial_head(nn.Layer):
    def __init__(self, model_config, compound_encoder):
        super(spatial_head, self).__init__()
        self.compound_encoder = compound_encoder
        self.hidden_size = model_config['hidden_size']
        self.dropout_rate = model_config['dropout_rate']
        self.act = model_config['act']
        self.pretrain_tasks = model_config['pretrain_tasks']
        # context mask:
        self.Cm_size = model_config['Cm_size']
        self.Cm_linear = nn.Linear(compound_encoder.embed_dim, self.Cm_size + 3)
        self.Cm_loss = nn.CrossEntropyLoss()
        # functional group
        self.Fg_linear = nn.Linear(compound_encoder.embed_dim, model_config['Fg_size'])
        self.Fg_loss = nn.BCEWithLogitsLoss()

        # bond length head
        self.Bl_mlp = MLP(2,
                          hidden_size=self.hidden_size,
                          act=self.act,
                          in_size=compound_encoder.embed_dim * 2,
                          out_size=1,
                          dropout_rate=self.dropout_rate)
        self.Bl_loss = nn.SmoothL1Loss()
        # atom distance head
        self.Ad_size = model_config['Ad_size']
        self.Ad_mlp = MLP(2,
                          hidden_size=self.hidden_size,
                          in_size=self.compound_encoder.embed_dim * 2,
                          act=self.act,
                          out_size=self.Ad_size + 3,
                          dropout_rate=self.dropout_rate)
        self.Ad_loss = nn.CrossEntropyLoss()

        # bond angle head
        self.Ba_mlp = MLP(2,
                          hidden_size=self.hidden_size,
                          act=self.act,
                          in_size=compound_encoder.embed_dim * 3,
                          out_size=1,
                          dropout_rate=self.dropout_rate)
        self.Ba_loss = nn.SmoothL1Loss()

        # dihedral angle head
        self.Da_mlp = MLP(2,
                          hidden_size=self.hidden_size,
                          act=self.act,
                          in_size=compound_encoder.embed_dim * 4,
                          out_size=1,
                          dropout_rate=self.dropout_rate)
        self.Da_loss = nn.SmoothL1Loss()

        #################
        self.freeze_config = {
            'conj_layers': False,
            'bond_angle_layers': False,
            'freeze_embedding': False
        }

    def freeze_conj_only(self):
        """freeze conj block"""
        if hasattr(self.compound_encoder, 'freeze_conj_only'):
            self.compound_encoder.freeze_conj_only()
            self.freeze_config.update({'conj_layers': True, 'bond_angle_layers': False})
        else:
            print("wrnning: compound_encoder not supprot freeze_conj_only")

    def freeze_bond_angle_only(self):
        """freeze spatial block"""
        if hasattr(self.compound_encoder, 'freeze_bond_angle_only'):
            self.compound_encoder.freeze_bond_angle_only()
            self.freeze_config.update({'conj_layers': False, 'bond_angle_layers': True})
        else:
            print("wrnning: compound_encoder not supprot freeze_bond_angle_only")

    def freeze_both(self):
        """freeze conj block"""
        if hasattr(self.compound_encoder, 'freeze_both'):
            self.compound_encoder.freeze_both()
            self.freeze_config.update({'conj_layers': True, 'bond_angle_layers': True})
        else:
            print("wrnning: compound_encoder not supprot freeze_both")

    def unfreeze_all(self):
        """unfreeze all"""
        if hasattr(self.compound_encoder, 'unfreeze_all'):
            self.compound_encoder.unfreeze_all()
            self.freeze_config.update({'conj_layers': False, 'bond_angle_layers': False})
        else:
            print("wrnning: compound_encoder not supprot unfreeze_all")

    def get_freezing_status(self):
        """gain freezing status"""
        if hasattr(self.compound_encoder, 'get_freezing_status'):
            status = self.compound_encoder.get_freezing_status()
            status['geo_pred_config'] = self.freeze_config.copy()
            return status
        else:
            return {
                'geo_pred_config': self.freeze_config.copy(),
                'message': 'encoder not support gain status'
            }

    def _get_Cm_loss(self, label_dict, node_repr):
        masked_node_repr = paddle.gather(node_repr, label_dict['Cm_node_i'])
        logits = self.Cm_linear(masked_node_repr)
        loss = self.Cm_loss(logits, label_dict['Cm_context_id'])
        return loss

    def _get_Fg_loss(self, label_dict, graph_repr):
        fg_label = paddle.concat(
                [label_dict['Fg_morgan'],
                label_dict['Fg_daylight'],
                label_dict['Fg_maccs']], 1)
        logits = self.Fg_linear(graph_repr)
        loss = self.Fg_loss(logits, fg_label)
        return loss

    def _get_Ba_loss(self, label_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, label_dict['Ba_node_i'])
        node_j_repr = paddle.gather(node_repr, label_dict['Ba_node_j'])
        node_k_repr = paddle.gather(node_repr, label_dict['Ba_node_k'])
        node_ijk_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr], 1)
        pred = self.Ba_mlp(node_ijk_repr)
        loss = self.Ba_loss(pred, label_dict['Ba_bond_angle'] / np.pi)
        return loss

    def _get_Bl_loss(self, label_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, label_dict['Bl_node_i'])
        node_j_repr = paddle.gather(node_repr, label_dict['Bl_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        pred = self.Bl_mlp(node_ij_repr)
        loss = self.Bl_loss(pred, label_dict['Bl_bond_length'])
        return loss

    def _get_Ad_loss(self, label_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, label_dict['Ad_node_i'])
        node_j_repr = paddle.gather(node_repr, label_dict['Ad_node_j'])
        node_ij_repr = paddle.concat([node_i_repr, node_j_repr], 1)
        logits = self.Ad_mlp.forward(node_ij_repr)
        atom_dist = paddle.clip(label_dict['Ad_atom_dist'], 0.0, 20.0)
        atom_dist_id = paddle.cast(atom_dist / 20.0 * self.Ad_size, 'int64')
        loss = self.Ad_loss(logits, atom_dist_id)
        return loss

    def _get_Dar_loss(self, label_dict, node_repr):
        node_i_repr = paddle.gather(node_repr, label_dict['Da_node_i'])
        node_j_repr = paddle.gather(node_repr, label_dict['Da_node_j'])
        node_k_repr = paddle.gather(node_repr, label_dict['Da_node_k'])
        node_l_repr = paddle.gather(node_repr, label_dict['Da_node_l'])

        node_ijkl_repr = paddle.concat([node_i_repr, node_j_repr, node_k_repr, node_l_repr], 1)
        pred = self.Da_mlp(node_ijkl_repr)
        loss = self.Da_loss(pred, label_dict['Da_dihedral_angle'] / np.pi)
        return loss

    def forward(self, graph_dict, label_dict, return_subloss=False):
        """
        Build the loss.
        """

        node_repr, edge_repr, plane_repr, graph_repr = self.compound_encoder.forward(
                graph_dict['atom_bond_graph'], graph_dict['bond_angle_graph'], graph_dict['dihedral_angle_graph'], graph_dict['conj_graph'])
        masked_node_repr, masked_edge_repr, masked_plane_repr, masked_graph_repr = self.compound_encoder.forward(
                graph_dict['masked_atom_bond_graph'], graph_dict['masked_bond_angle_graph'], graph_dict['masked_dihedral_angle_graph'], graph_dict['conj_graph'])

        subject_losses = {}
        subject_losses['Cm_loss'] = self._get_Cm_loss(label_dict, node_repr)
        subject_losses['Cm_loss'] += self._get_Cm_loss(label_dict, masked_node_repr)

        subject_losses['Fg_loss'] = self._get_Fg_loss(label_dict, graph_repr)
        subject_losses['Fg_loss'] += self._get_Fg_loss(label_dict, masked_graph_repr)

        subject_losses['Bl_loss'] = self._get_Bl_loss(label_dict, node_repr)
        subject_losses['Bl_loss'] += self._get_Bl_loss(label_dict, masked_node_repr)

        subject_losses['Ad_loss'] = self._get_Ad_loss(label_dict, node_repr)
        subject_losses['Ad_loss'] += self._get_Ad_loss(label_dict, masked_node_repr)

        subject_losses['Ba_loss'] = self._get_Ba_loss(label_dict, node_repr)
        subject_losses['Ba_loss'] += self._get_Ba_loss(label_dict, masked_node_repr)

        subject_losses['Dar_loss'] = self._get_Dar_loss(label_dict, node_repr)
        subject_losses['Dar_loss'] += self._get_Dar_loss(label_dict, masked_node_repr)

        loss = 0
        for name in subject_losses:
            loss += subject_losses[name]
        if return_subloss:
            return loss, subject_losses
        else:
            return loss