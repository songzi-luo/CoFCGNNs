

"""
| Featurizers for pretrain-gnn.

| Adapted from https://github.com/snap-stanford/pretrain-gnns/tree/master/chem/utils.py
"""

import numpy as np
import networkx as nx
from copy import deepcopy
import pgl
from rdkit.Chem import AllChem
from src.dihedral_angle_graph import add_diangle
from sklearn.metrics import pairwise_distances
import hashlib
from utils.compound_tools import *



def md5_hash(string):
    """tbd"""
    md5 = hashlib.md5(string.encode('utf-8')).hexdigest()
    return int(md5, 16)


def mask_context_of_geognn_graph(
        ab_g,
        ba_g,
        da_g,
        co_g,
        edges,
        conj_edge,
        plane_atoms,
        target_atom_indices=None, 
        mask_ratio=None, 
        mask_value=0, 
        subgraph_num=None):

    """tbd"""
    # print('      call mask_context_of_geognn_graph')

    def get_subgraph_str(ab_g, da_g, atom_index, nei_atom_indices, nei_bond_indices, nei_plane_indices):
        """tbd"""
        atomic_num = ab_g.node_feat['atomic_num'].flatten()  # flatten() convert into one-dimension array
        bond_type = ab_g.edge_feat['bond_type'].flatten()
        plane_in_ring = da_g.node_feat['plane_in_ring'].flatten()
        subgraph_str = 'A' + str(atomic_num[atom_index])
        subgraph_str += 'N' + ':'.join([str(x) for x in np.sort(atomic_num[nei_atom_indices])])
        subgraph_str += 'E' + ':'.join([str(x) for x in np.sort(bond_type[nei_bond_indices])])
        subgraph_str += 'P' + ':'.join([str(x) for x in np.sort(plane_in_ring[nei_plane_indices])])
        return subgraph_str

    ab_g = deepcopy(ab_g)
    co_g = deepcopy(co_g)
    N = ab_g.num_nodes
    E = ba_g.num_nodes
    P = da_g.num_nodes
    C = co_g.num_edges
    full_conj_indices = np.arange(C)
    full_atom_indices = np.arange(N)
    full_bond_indices = np.arange(E)
    full_plane_indices = np.arange(P)

    """
    Randomly generate atom indices for masking
    """
    if target_atom_indices is None:
        masked_size = max(1, int(N * mask_ratio))   # at least 1 atom will be selected.
        target_atom_indices = np.random.choice(full_atom_indices, size=masked_size, replace=False)

    target_labels = []
    Cm_node_i = []
    masked_bond_indices = []
    masked_plane_indices = []
    masked_conj_indices = []

    """
    Find Cm_node_i, masked_bond_indices, masked_plane_indices, target_labels
    Cm_node_i: target masked atom indices, neighbour atom indices of target masked atom
    masked_bond_indices: bond indices associated with target masked atom
    masked_plane_indices: plane indices associated with target masked atom
    """
    for atom_index in target_atom_indices:

        """
        Find neighbour bond indices based on target atom index
        """
        ab_left_edge_indices = full_bond_indices[edges[:, 0] == atom_index]
        ab_right_edge_indices = full_bond_indices[edges[:, 1] == atom_index]
        co_left_edge_indices = full_conj_indices[conj_edge[:, 0] == atom_index]
        co_right_edge_indices = full_conj_indices[conj_edge[:, 1] == atom_index]
        co_left_edge_indices = np.random.choice(co_left_edge_indices,
                                                size=int(np.ceil(len(co_left_edge_indices) * 0.5)), replace=False)
        co_right_edge_indices = np.random.choice(co_right_edge_indices,
                                                 size=int(np.ceil(len(co_right_edge_indices) * 0.5)), replace=False)

        ab_left_edge_indices = np.random.choice(ab_left_edge_indices, size=int(np.ceil(len(ab_left_edge_indices) * 0.5)), replace=False)
        ab_right_edge_indices = np.random.choice(ab_right_edge_indices, size=int(np.ceil(len(ab_right_edge_indices) * 0.5)), replace=False)
        ab_nei_bond_indices = np.append(ab_left_edge_indices, ab_right_edge_indices)  # find all bound that contain target atom
        co_nei_bond_indices = np.append(co_left_edge_indices, co_right_edge_indices)


        """
        Find all neighbour atom in nei_bond_indices of target atom
        """
        left_nei_atom_indices = ab_g.edges[ab_left_edge_indices, 1]
        right_nei_atom_indices = ab_g.edges[ab_right_edge_indices, 0]
        nei_atom_indices = np.append(left_nei_atom_indices, right_nei_atom_indices)  # find all neighbour of target atom


        """
        Find plane indices contain target atom index
        """
        a0_nei_plane_indices = full_plane_indices[plane_atoms[:, 0] == atom_index]
        a1_nei_plane_indices = full_plane_indices[plane_atoms[:, 1] == atom_index]
        b0_nei_plane_indices = full_plane_indices[plane_atoms[:, 2] == atom_index]
        b1_nei_plane_indices = full_plane_indices[plane_atoms[:, 3] == atom_index]
        nei_plane_indices = np.append(a0_nei_plane_indices, a1_nei_plane_indices)
        nei_plane_indices = np.append(nei_plane_indices, b0_nei_plane_indices)
        nei_plane_indices = np.append(nei_plane_indices, b1_nei_plane_indices)
        nei_plane_indices = np.random.choice(nei_plane_indices, size=int(np.ceil(len(nei_plane_indices) * 0.5)), replace=False)

        """
        Generate graph str for target atom based on atom_index, nei_atom_indices, nei_bond_indices
        """

        subgraph_str = get_subgraph_str(ab_g, da_g, atom_index, nei_atom_indices, ab_nei_bond_indices, nei_plane_indices)
        subgraph_id = md5_hash(subgraph_str) % subgraph_num
        target_label = subgraph_id



        """
        Respectively construct lists of target masked atoms, neighbour atoms, neighbour bonds, labels
        """
        Cm_node_i.append([atom_index])  # list of target masked atom indices
        Cm_node_i.append(nei_atom_indices)  # add neighbour atom indices for each target masked atom to the above list
        masked_bond_indices.append(ab_nei_bond_indices)  # lists of neighbour bond indices for each target masked atom
        masked_plane_indices.append(nei_plane_indices)   # lists of neighbour plane indices for each target masked atom
        masked_conj_indices.append(co_nei_bond_indices)
        target_labels.append(target_label)

    target_atom_indices = np.array(target_atom_indices)  # convert to array
    Cm_node_i = np.concatenate(Cm_node_i, 0)  # convert to N*1 array
    masked_bond_indices = np.concatenate(masked_bond_indices, 0)  # convert to N*1 array
    masked_plane_indices = np.concatenate(masked_plane_indices, 0)  # convert to N*1 array
    masked_co_bond_indices = np.concatenate(masked_conj_indices, 0)
    target_labels = np.array(target_labels)  # convert to array


    """
    =====================
    ab_g co_g mask
    =====================
    """
    for name in ab_g.node_feat:
        ab_g.node_feat[name][Cm_node_i] = mask_value
    for name in ab_g.edge_feat:
        ab_g.edge_feat[name][masked_bond_indices] = mask_value
    for name in co_g.edge_feat:
        co_g.edge_feat[name][masked_co_bond_indices] = mask_value


    full_superedge_indices = np.arange(ba_g.num_edges)
    masked_superedge_indices = []
    for bond_index in masked_bond_indices:
        left_indices = full_superedge_indices[ba_g.edges[:, 0] == bond_index]
        right_indices = full_superedge_indices[ba_g.edges[:, 1] == bond_index]
        masked_superedge_indices.append(np.append(left_indices, right_indices))
    masked_superedge_indices = np.concatenate(masked_superedge_indices, 0)
    for name in ba_g.edge_feat:
        ba_g.edge_feat[name][masked_superedge_indices] = mask_value
    return [ab_g, ba_g, da_g,co_g, target_atom_indices, target_labels]
    

def get_pretrain_bond_angle(edges, atom_poses):
    def _get_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2 + 1e-5)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle
    def _add_item(
            node_i_indices, node_j_indices, node_k_indices, bond_angles,
            node_i_index, node_j_index, node_k_index):
        node_i_indices += [node_i_index, node_k_index]
        node_j_indices += [node_j_index, node_j_index]
        node_k_indices += [node_k_index, node_i_index]
        pos_i = atom_poses[node_i_index]
        pos_j = atom_poses[node_j_index]
        pos_k = atom_poses[node_k_index]
        angle = _get_angle(pos_i - pos_j, pos_k - pos_j)
        bond_angles += [angle, angle]

    E = len(edges)
    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    bond_angles = []
    for edge_i in range(E - 1):
        for edge_j in range(edge_i + 1, E):
            a0, a1 = edges[edge_i]
            b0, b1 = edges[edge_j]
            if a0 == b0 and a1 == b1:
                continue
            if a0 == b1 and a1 == b0:
                continue
            if a0 == b0:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b1)
            if a0 == b1:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a1, a0, b0)
            if a1 == b0:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b1)
            if a1 == b1:
                _add_item(
                        node_i_indices, node_j_indices, node_k_indices, bond_angles,
                        a0, a1, b0)
    node_ijk = np.array([node_i_indices, node_j_indices, node_k_indices])
    uniq_node_ijk, uniq_index = np.unique(node_ijk, return_index=True, axis=1)
    node_i_indices, node_j_indices, node_k_indices = uniq_node_ijk
    bond_angles = np.array(bond_angles)[uniq_index]
    return [node_i_indices, node_j_indices, node_k_indices, bond_angles]

# def get_pretrain_dih_angle(edges, atom_poses):
#     """tbd"""
#     def _get_angle(vec1, vec2):
#         norm1 = np.linalg.norm(vec1)
#         norm2 = np.linalg.norm(vec2)
#         if norm1 == 0 or norm2 == 0:
#             return 0
#         vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
#         vec2 = vec2 / (norm2 + 1e-5)
#         angle = np.arccos(np.dot(vec1, vec2))
#         return angle
#     def _add_item(
#             node_i_indices, node_j_indices, node_k_indices, node_l_indices, bond_angles,
#             node_i_index, node_j_index, node_k_index, node_l_index):
#         node_i_indices += [node_i_index, node_k_index]
#         node_j_indices += [node_j_index, node_j_index]
#         node_k_indices += [node_k_index, node_i_index]
#         node_l_indices += [node_l_index, node_i_index]
#         pos_i = atom_poses[node_i_index]
#         pos_j = atom_poses[node_j_index]
#         pos_k = atom_poses[node_k_index]
#         pos_l = atom_poses[node_l_index]
#         angle = _get_angle(pos_i - pos_j, pos_k - pos_l)
#         bond_angles += [angle, angle]
#
#     E = len(edges)
#     node_i_indices = []
#     node_j_indices = []
#     node_k_indices = []
#     node_l_indices = []
#     bond_angles = []
#     for edge_i in range(E - 1):
#         for edge_j in range(edge_i + 1, E):
#             a0, a1 = edges[edge_i]
#             b0, b1 = edges[edge_j]
#             if a0 == b0 and a1 == b1:  # [0, 1], [0, 1]
#                 continue
#             if a0 == b1 and a1 == b0:  # [0, 1], [1, 0]
#                 continue
#             if a0 == b0:  # [1, 0], [1, 2] => 0-1-2
#                 continue
#             if a0 == b1:
#                 continue
#             if a1 == b0:
#                 continue
#             if a1 == b1:
#                 continue
#             if a0 == b1+1:
#                 _add_item(node_i_indices, node_j_indices, node_k_indices,node_l_indices, bond_angles,a1, a0, b1, b0)
#             if a0 == b0+1:  # [1, 0], [2, 1] => 0-1-2
#                 _add_item(
#                         node_i_indices, node_j_indices, node_k_indices, node_l_indices,bond_angles,
#                         a1, a0, b0, b1)
#             if a1 == b0-1:  # [1, 2], [2, 3] => 1-2-3
#                 _add_item(
#                         node_i_indices, node_j_indices, node_k_indices, node_l_indices,bond_angles,
#                         a0, a1, b1, b0)
#             if a1 == b1-1:  # [1, 2], [3, 2] => 1-2-3
#                 _add_item(
#                         node_i_indices, node_j_indices, node_k_indices, node_l_indices,bond_angles,
#                         a0, a1, b0, b1)
#     node_ijkl = np.array([node_i_indices, node_j_indices, node_k_indices])
#     uniq_node_ijkl, uniq_index = np.unique(node_ijkl, return_index=True, axis=1)  # order node_i_indices
#     node_i_indices, node_j_indices, node_k_indices,node_l_indices = uniq_node_ijkl
#     bond_angles = np.array(bond_angles)[uniq_index]
#     return [node_i_indices, node_j_indices, node_k_indices,node_l_indices, bond_angles]


import numpy as np
from collections import defaultdict

def get_pretrain_dih_angle_corrected(edges, atom_poses):

    # construct adj table
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # calculat angle ij-kl
    def _calculate_dihedral(pos_i, pos_j, pos_k, pos_l):
        # ij, jk, kl
        vec_ij = pos_i - pos_j
        vec_jk = pos_j - pos_k
        vec_kl = pos_k - pos_l

        # n1: mormal vector-ijk ，n: mormal vector-jkl
        n1 = np.cross(vec_ij, vec_jk)
        n2 = np.cross(vec_jk, vec_kl)


        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        if norm_n1 < 1e-6 or norm_n2 < 1e-6:
            return 0.0

        n1 = n1 / norm_n1
        n2 = n2 / norm_n2
        cos_angle = np.dot(n1, n2)
        #  [-π, π]
        sin_angle = np.dot(np.cross(n1, n2), vec_jk) / (np.linalg.norm(vec_jk) + 1e-8)
        angle = np.arctan2(sin_angle, cos_angle)

        return angle

    node_i_indices = []
    node_j_indices = []
    node_k_indices = []
    node_l_indices = []
    dihedral_angles = []

    # traverse all continual 4 atoms i-j-k-l
    for j in list(adj.keys()):
        for k in adj[j]:
            for i in adj[j]:
                if i == k:
                    continue
                for l in adj[k]:
                    if l == j:
                        continue
                    angle = _calculate_dihedral(
                        atom_poses[i], atom_poses[j],
                        atom_poses[k], atom_poses[l]
                    )
                    node_i_indices.append(i)
                    node_j_indices.append(j)
                    node_k_indices.append(k)
                    node_l_indices.append(l)
                    dihedral_angles.append(angle)
    if len(node_i_indices) > 0:
        node_ijkl = np.array([node_i_indices, node_j_indices, node_k_indices, node_l_indices])
        uniq_node_ijkl, uniq_index = np.unique(node_ijkl, axis=1, return_index=True)
        i_idx, j_idx, k_idx, l_idx = uniq_node_ijkl
        dihedral_angles = np.array(dihedral_angles)[uniq_index]
    else:
        i_idx, j_idx, k_idx, l_idx, dihedral_angles = [], [], [], [], []

    return [i_idx, j_idx, k_idx, l_idx, dihedral_angles]


class PredTransformFn(object):
    """features for down tasks"""
    def __init__(self, pretrain_tasks, mask_ratio):
        self.pretrain_tasks = pretrain_tasks
        self.mask_ratio = mask_ratio
        self.gen_new_data_with_dihedral_angle_simple = add_diangle

    def prepare_pretrain_task(self, data):
        """
        prepare data for pretrain task
        """
        # calculate angle between two edges based on atom_pos
        node_i, node_j, node_k, bond_angles = get_pretrain_bond_angle(data['edges'], data['atom_pos'])  # atom_pos is the coordinate
        dnode_i, dnode_j, dnode_k,dnode_l, di_angles = get_pretrain_dih_angle_corrected(data['edges'], data['atom_pos'])
        # angle among Ba_node_i, Ba_node_j, Ba_node_k
        data['Ba_node_i'] = node_i
        data['Ba_node_j'] = node_j
        data['Ba_node_k'] = node_k
        data['Ba_bond_angle'] = bond_angles

        # get bond length between Bl_node_i and Bl_node_j
        data['Bl_node_i'] = np.array(data['edges'][:, 0])
        data['Bl_node_j'] = np.array(data['edges'][:, 1])
        data['Bl_bond_length'] = np.array(data['bond_length'])

        # calculate distance between two atoms
        n = len(data['atom_pos'])
        dist_matrix = pairwise_distances(data['atom_pos'])
        indice = np.repeat(np.arange(n).reshape([-1, 1]), n, axis=1)
        data['Ad_node_i'] = indice.reshape([-1, 1])
        data['Ad_node_j'] = indice.T.reshape([-1, 1])
        data['Ad_atom_dist'] = dist_matrix.reshape([-1, 1])

        data['Da_node_i'] = dnode_i
        data['Da_node_j'] = dnode_j
        data['Da_node_k'] = dnode_k
        data['Da_node_l'] = dnode_l
        data['Da_dihedral_angle'] = di_angles

        return data

    def __call__(self, raw_data):
        """
        smiles return to a single graph data.
        Args:
            raw_data: smiles
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data
        mol = AllChem.MolFromSmiles(smiles)

        if mol is None:
            return None
        data = mol_to_graph(mol)
        data['smiles'] = smiles
        data = self.prepare_pretrain_task(data)

        if len(data['Da_node_i']) == 0:
            return None

        data = self.gen_new_data_with_dihedral_angle_simple(data)
        return data


class PredCollateFn(object):
    """tbd"""
    def __init__(self, MLP_head_config, encoder_config):
        self.atom_names = encoder_config["atom_names"]
        self.bond_names = encoder_config["bond_names"]
        self.bond_float_names = encoder_config["bond_float_names"]
        self.bond_angle_float_names = encoder_config["bond_angle_float_names"]
        self.plane_names = encoder_config["plane_names"]
        self.plane_float_names = encoder_config["plane_float_names"]
        self.dihedral_angle_float_names = encoder_config["dihedral_angle_float_names"]
        self.pretrain_tasks = MLP_head_config["pretrain_tasks"]
        self.mask_ratio = MLP_head_config["mask_ratio"]
        self.Cm_size = MLP_head_config["Cm_size"]
        self.conj_name = ['all_arom', 'hemi_arom', 'arom_conj', 'no_arom']

        print('[PredCollateFn] atom_names:%s' % self.atom_names)
        print('[PredCollateFn] bond_names:%s' % self.bond_names)
        print('[PredCollateFn] bond_float_names:%s' % self.bond_float_names)
        print('[PredCollateFn] bond_angle_float_names:%s' % self.bond_angle_float_names)
        print('[PredCollateFn] plane_names:%s' % self.plane_names)
        print('[PredCollateFn] plane_float_names:%s' % self.plane_float_names)
        print('[PredCollateFn] dihedral_angle_float_names:%s' % self.dihedral_angle_float_names)
        print('[PredCollateFn] pretrain_tasks:%s' % self.pretrain_tasks)
        print('[PredCollateFn] mask_ratio:%s' % self.mask_ratio)
        print('[PredCollateFn] Cm_size:%s' % self.Cm_size)


        
    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def __call__(self, batch_data_list):

        # graph list
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        dihedral_angle_graph_list = []
        conj_graph_list = []
        masked_atom_bond_graph_list = []
        masked_bond_angle_graph_list = []
        masked_dihedral_angle_graph_list = []
        masked_conj_graph_list = []
        homo = []
        lumo = []
        dipole = []
        mulliken_charges = []


        Cm_node_i = []
        Cm_context_id = []

        # fingerprint
        Fg_morgan = []
        Fg_daylight = []
        Fg_maccs = []

        # bond angle for three nodes, two bond, one angle
        Ba_node_i = []
        Ba_node_j = []
        Ba_node_k = []
        Ba_bond_angle = []

        # bond length for two nodes, one bond, one length
        Bl_node_i = []
        Bl_node_j = []
        Bl_bond_length = []

        # atom distance for two nodes, one distance
        Ad_node_i = []
        Ad_node_j = []
        Ad_atom_dist = []

        # angle for four nodes, two planes share on edge, on angle
        Da_node_i = []
        Da_node_k = []
        Da_node_l = []
        Da_node_j = []
        Da_dihedral_angle = []


        node_count = 0
        ii = 0

        for data in batch_data_list:
            ii += 1
            N = len(data[self.atom_names[0]])
            E = len(data['edges'])
            P = len(data['BondAngleGraph_edges'])

            ab_g = pgl.graph.Graph(
                    num_nodes=N,
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            ba_g = pgl.graph.Graph(
                    num_nodes=E,
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            da_g = pgl.graph.Graph(
                    num_nodes=P,
                    edges=data['DihedralAngleGraph_edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.plane_names + self.plane_float_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.dihedral_angle_float_names})
            co_g = pgl.Graph(num_nodes=N,
                             edges=data['conj_edge'],
                             node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                             edge_feat={name: data[name].reshape([-1, 1]) for name in self.conj_name})
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            dihedral_angle_graph_list.append(da_g)
            conj_graph_list.append(co_g)

            """
            Contruct mased_ab_g, masked_ba_g, masked_da_g,masked_co_g, mask_node_i, context_id
            """
            edges = data['edges']
            plane_atoms = data['plane_atoms']
            conj_edge = data['conj_edge']
            masked_ab_g, masked_ba_g, masked_da_g,masked_co_g, mask_node_i, context_id = mask_context_of_geognn_graph(
                    ab_g,
                    ba_g,
                    da_g,
                    co_g,
                    edges,
                    conj_edge,
                    plane_atoms,
                    mask_ratio=self.mask_ratio,
                    subgraph_num= self.Cm_size)
            masked_atom_bond_graph_list.append(masked_ab_g)
            masked_bond_angle_graph_list.append(masked_ba_g)
            masked_dihedral_angle_graph_list.append(masked_da_g)
            masked_conj_graph_list.append(masked_co_g)
            homo.append(data['homo'])
            lumo.append(data['lumo'])
            dipole.append(data['dipole'])
            mulliken_charges.append(data['mulliken_charges'])

            Cm_node_i.append(mask_node_i + node_count)
            Cm_context_id.append(context_id)

            Fg_morgan.append(data['morgan_fp'])
            Fg_daylight.append(data['daylight_fg_counts'])
            Fg_maccs.append(data['maccs_fp'])

            Ba_node_i.append(data['Ba_node_i'] + node_count)
            Ba_node_j.append(data['Ba_node_j'] + node_count)
            Ba_node_k.append(data['Ba_node_k'] + node_count)
            Ba_bond_angle.append(data['Ba_bond_angle'])

            Bl_node_i.append(data['Bl_node_i'] + node_count)
            Bl_node_j.append(data['Bl_node_j'] + node_count)
            Bl_bond_length.append(data['Bl_bond_length'])

            Ad_node_i.append(data['Ad_node_i'] + node_count)
            Ad_node_j.append(data['Ad_node_j'] + node_count)
            Ad_atom_dist.append(data['Ad_atom_dist'])

            Da_node_i.append(data['Da_node_i'] + node_count)
            Da_node_k.append(data['Da_node_k'] + node_count)
            Da_node_l.append(data['Da_node_l'] + node_count)
            Da_node_j.append(data['Da_node_j'] + node_count)
            Da_dihedral_angle.append(data['Da_dihedral_angle'])

            node_count += N

        graph_dict = {}    
        label_dict = {}

        """
        atom-bond graph graph_dict, label_dict
        """
        # print('  construct graph dict')
        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        graph_dict['atom_bond_graph'] = atom_bond_graph


        """
        bond-angle graph graph_dict, label_dict
        """
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        graph_dict['bond_angle_graph'] = bond_angle_graph

        """
        dihedral-angle graph graph_dict, label_dict
        """
        dihedral_angle_graph = pgl.Graph.batch(dihedral_angle_graph_list)
        self._flat_shapes(dihedral_angle_graph.node_feat)
        self._flat_shapes(dihedral_angle_graph.edge_feat)
        graph_dict['dihedral_angle_graph'] = dihedral_angle_graph

        """
        conjugate graph graph_dict, label_dict
        """
        conj_graph = pgl.Graph.batch(conj_graph_list)
        self._flat_shapes(conj_graph.node_feat)
        self._flat_shapes(conj_graph.edge_feat)
        graph_dict['conj_graph'] = conj_graph

        """
        masked atom-bond graph graph_dict, label_dict
        """
        masked_atom_bond_graph = pgl.Graph.batch(masked_atom_bond_graph_list)
        self._flat_shapes(masked_atom_bond_graph.node_feat)
        self._flat_shapes(masked_atom_bond_graph.edge_feat)
        graph_dict['masked_atom_bond_graph'] = masked_atom_bond_graph

        """
        masked bond-angle graph graph_dict, label_dict
        """
        masked_bond_angle_graph = pgl.Graph.batch(masked_bond_angle_graph_list)
        self._flat_shapes(masked_bond_angle_graph.node_feat)
        self._flat_shapes(masked_bond_angle_graph.edge_feat)
        graph_dict['masked_bond_angle_graph'] = masked_bond_angle_graph


        """
        masked dihedral-angle graph graph_dict, label_dict
        """
        masked_dihedral_angle_graph = pgl.Graph.batch(masked_dihedral_angle_graph_list)
        self._flat_shapes(masked_dihedral_angle_graph.node_feat)
        self._flat_shapes(masked_dihedral_angle_graph.edge_feat)
        graph_dict['masked_dihedral_angle_graph'] = masked_dihedral_angle_graph

        """
        masked conjugate graph graph_dict, label_dict
        """
        masked_conj_graph = pgl.Graph.batch(masked_conj_graph_list)
        self._flat_shapes(masked_conj_graph.node_feat)
        self._flat_shapes(masked_conj_graph.edge_feat)
        graph_dict['masked_conj_graph'] = masked_conj_graph


        label_dict['Cm_node_i'] = np.concatenate(Cm_node_i, 0).reshape(-1).astype('int64')
        label_dict['Cm_context_id'] = np.concatenate(Cm_context_id, 0).reshape(-1, 1).astype('int64')

        label_dict['Fg_morgan'] = np.array(Fg_morgan, 'float32')
        label_dict['Fg_daylight'] = (np.array(Fg_daylight) > 0).astype('float32')
        label_dict['Fg_maccs'] = np.array(Fg_maccs, 'float32')

        label_dict['Ba_node_i'] = np.concatenate(Ba_node_i, 0).reshape(-1).astype('int64')
        label_dict['Ba_node_j'] = np.concatenate(Ba_node_j, 0).reshape(-1).astype('int64')
        label_dict['Ba_node_k'] = np.concatenate(Ba_node_k, 0).reshape(-1).astype('int64')
        label_dict['Ba_bond_angle'] = np.concatenate(Ba_bond_angle, 0).reshape(-1, 1).astype('float32')

        label_dict['Bl_node_i'] = np.concatenate(Bl_node_i, 0).reshape(-1).astype('int64')
        label_dict['Bl_node_j'] = np.concatenate(Bl_node_j, 0).reshape(-1).astype('int64')
        label_dict['Bl_bond_length'] = np.concatenate(Bl_bond_length, 0).reshape(-1, 1).astype('float32')

        label_dict['Ad_node_i'] = np.concatenate(Ad_node_i, 0).reshape(-1).astype('int64')
        label_dict['Ad_node_j'] = np.concatenate(Ad_node_j, 0).reshape(-1).astype('int64')
        label_dict['Ad_atom_dist'] = np.concatenate(Ad_atom_dist, 0).reshape(-1, 1).astype('float32')

        label_dict['Da_node_i'] = np.concatenate(Da_node_i, 0).reshape(-1).astype('int64')
        label_dict['Da_node_k'] = np.concatenate(Da_node_k, 0).reshape(-1).astype('int64')
        label_dict['Da_node_l'] = np.concatenate(Da_node_l, 0).reshape(-1).astype('int64')
        label_dict['Da_node_j'] = np.concatenate(Da_node_j, 0).reshape(-1).astype('int64')
        label_dict['Da_dihedral_angle'] = np.concatenate(Da_dihedral_angle, 0).reshape(-1, 1).astype('float32')

        label_dict['homo'] = np.concatenate(homo, 0).reshape(-1, 1).astype('float32')
        label_dict['lumo'] = np.concatenate(lumo, 0).reshape(-1, 1).astype('float32')
        label_dict['dipole'] = np.concatenate(dipole, 0).reshape(-1, 1).astype('float32')
        label_dict['mulliken_charges'] = np.concatenate(mulliken_charges, 0).reshape(-1, 1).astype('float32')

        return graph_dict, label_dict

