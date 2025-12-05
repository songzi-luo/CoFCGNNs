"""
downstream featurizer
"""

import numpy as np
import pgl
from rdkit.Chem import AllChem
from utils.compound_tools import mol_to_graph
from src.dihedral_angle_graph import add_diangle


class DownstreamTransformFn(object):
    """Features for downstream model"""
    def __init__(self, is_inference=False):
        self.is_inference = is_inference

    def __call__(self, raw_data):
        """
        Features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles = raw_data['smiles']
        print(smiles)
        if smiles is None:
            return None
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            return None
        data = mol_to_graph(mol)
        if not self.is_inference:
            data['label'] = raw_data['label'].reshape([-1])
        data['smiles'] = smiles
        return data


class DownstreamCollateFn(object):
    """CollateFn for downstream model"""
    def __init__(self, 
            atom_names, 
            bond_names, 
            bond_float_names,
            bond_angle_float_names,
            plane_names,
            plane_float_names,
            dihedral_angle_float_names,
            task_type,
            is_inference=False):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names
        self.plane_names = plane_names
        self.plane_float_names = plane_float_names
        self.dihedral_angle_float_names = dihedral_angle_float_names
        self.task_type = task_type
        self.is_inference = is_inference
        self.conj_name = ['all_arom', 'hemi_arom', 'arom_conj', 'no_arom']

    def _flat_shapes(self, di):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in di:
            di[name] = di[name].reshape([-1])
    
    def __call__(self, data_list):
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        dihedral_angle_graph_list = []
        label_list = []
        conj_graph_list = []
        for data in data_list:
            ab_g = pgl.Graph(
                    num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            ba_g = pgl.Graph(
                    num_nodes=len(data['edges']),
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            da_g = pgl.graph.Graph(
                num_nodes=len(data['BondAngleGraph_edges']),
                edges=data['DihedralAngleGraph_edges'],
                node_feat={name: data[name].reshape([-1, 1]) for name in self.plane_names + self.plane_float_names},
                edge_feat={name: data[name].reshape([-1, 1]) for name in self.dihedral_angle_float_names})
            co_g = pgl.Graph(
                num_nodes=len(data[self.atom_names[0]]),
                edges=data['conj_edge'],
                node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                edge_feat={name: data[name].reshape([-1, 1]) for name in self.conj_name})
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)
            dihedral_angle_graph_list.append(da_g)
            conj_graph_list.append(co_g)
            if not self.is_inference:
                label_list.append(data['label'])
        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        conj_graph = pgl.Graph.batch(conj_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        dihedral_angle_graph = pgl.Graph.batch(dihedral_angle_graph_list)
        # TODO: reshape due to pgl limitations on the shape
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        self._flat_shapes(dihedral_angle_graph.node_feat)
        self._flat_shapes(dihedral_angle_graph.edge_feat)
        self._flat_shapes(conj_graph.edge_feat)
        self._flat_shapes(conj_graph.node_feat)
        if not self.is_inference:
            if self.task_type == 'class':
                labels = np.array(label_list)
                valids = (labels != 0.5)
                return [atom_bond_graph, bond_angle_graph, dihedral_angle_graph, conj_graph, valids, labels]
            else:
                labels = np.array(label_list, 'float32')
                return atom_bond_graph, bond_angle_graph, dihedral_angle_graph, conj_graph, labels
        else:
            return atom_bond_graph, bond_angle_graph, dihedral_angle_graph, conj_graph

