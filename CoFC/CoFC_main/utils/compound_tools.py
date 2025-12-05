"""
| Tools for compound features.
| Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
"""
import os
from collections import OrderedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdchem
from utils.compound_constants import DAY_LIGHT_FG_SMARTS_LIST
from src.dihedral_angle_graph import add_diangle



#
# def check_smiles_validity(smiles):
#     """
#     Check whether the smile can't be converted to rdkit mol object.
#     """
#     try:
#         m = Chem.MolFromSmiles(smiles)
#         if m:
#             return True
#         else:
#             return False
#     except Exception as e:
#         return False
#
#
# def split_rdkit_mol_obj(mol):
#     """
#     Split rdkit mol object containing multiple species or one species into a
#     list of mol objects or a list containing a single object respectively.
#
#     Args:
#         mol: rdkit mol object.
#     """
#     smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
#     smiles_list = smiles.split('.')
#     mol_species_list = []
#     for s in smiles_list:
#         if check_smiles_validity(s):
#             mol_species_list.append(AllChem.MolFromSmiles(s))
#     return mol_species_list


# def get_largest_mol(mol_list):
#     """
#     Given a list of rdkit mol objects, returns mol object containing the
#     largest num of atoms. If multiple containing largest num of atoms,
#     picks the first one.
#
#     Args:
#         mol_list(list): a list of rdkit mol object.
#
#     Returns:
#         the largest mol.
#     """
#     num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
#     largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
#     return mol_list[largest_mol_idx]

def rdchem_enum_to_list(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    return [values[i] for i in range(len(values))]


def safe_index(alist, elem):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return alist.index(elem)
    except ValueError:
        return len(alist) - 1


# def get_atom_feature_dims(list_acquired_feature_names):
#     """ tbd
#     """
#     return list(map(len, [CompoundKit.atom_vocab_dict[name] for name in list_acquired_feature_names]))


# def get_bond_feature_dims(list_acquired_feature_names):
#     """ tbd
#     """
#     list_bond_feat_dim = list(map(len, [CompoundKit.bond_vocab_dict[name] for name in list_acquired_feature_names]))
#     # +1 for self loop edges
#     return [_l + 1 for _l in list_bond_feat_dim]


class CompoundKit(object):
    """
    CompoundKit
    """
    atom_vocab_dict = {
        "atomic_num": list(range(1, 119)) + ['misc'],
        "chiral_tag": rdchem_enum_to_list(rdchem.ChiralType.values),
        "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "explicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        "hybridization": rdchem_enum_to_list(rdchem.HybridizationType.values),
        "implicit_valence": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'misc'],
        "is_aromatic": [0, 1],
        "total_numHs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'num_radical_e': [0, 1, 2, 3, 4, 'misc'],
        'atom_is_in_ring': [0, 1],
        'valence_out_shell': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size5': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size6': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size7': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'in_num_ring_with_size8': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    }
    bond_vocab_dict = {
        "bond_dir": rdchem_enum_to_list(rdchem.BondDir.values),
        "bond_type": rdchem_enum_to_list(rdchem.BondType.values),
        "is_in_ring": [0, 1],
        "all_arom": [0, 1], "hemi_arom": [0, 1], "arom_conj": [0, 1], "no_arom": [0, 1],
        'bond_stereo': rdchem_enum_to_list(rdchem.BondStereo.values),
        'is_conjugated': [0, 1],
    }
    # float features
    atom_float_names = ["van_der_waals_radis", "partial_charge", 'mass']
    # bond_float_feats= ["bond_length", "bond_angle"]     # optional

    plane_vocab_dict = {
        "plane_in_ring": list(range(1, 10)),
    }

    ### functional groups
    day_light_fg_smarts_list = DAY_LIGHT_FG_SMARTS_LIST
    day_light_fg_mo_list = [Chem.MolFromSmarts(smarts) for smarts in day_light_fg_smarts_list]

    morgan_fp_N = 200
    morgan2048_fp_N = 2048
    maccs_fp_N = 167

    period_table = Chem.GetPeriodicTable()

    ### atom

    @staticmethod
    def get_atom_value(atom, name):
        """get atom values"""
        if name == 'atomic_num':
            return atom.GetAtomicNum()
        elif name == 'chiral_tag':
            return atom.GetChiralTag()
        elif name == 'degree':
            return atom.GetDegree()
        elif name == 'explicit_valence':
            return atom.GetExplicitValence()
        elif name == 'formal_charge':
            return atom.GetFormalCharge()
        elif name == 'hybridization':
            return atom.GetHybridization()
        elif name == 'implicit_valence':
            return atom.GetImplicitValence()
        elif name == 'is_aromatic':
            return int(atom.GetIsAromatic())
        elif name == 'mass':
            return int(atom.GetMass())
        elif name == 'total_numHs':
            return atom.GetTotalNumHs()
        elif name == 'num_radical_e':
            return atom.GetNumRadicalElectrons()
        elif name == 'atom_is_in_ring':
            return int(atom.IsInRing())
        elif name == 'valence_out_shell':
            return CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())
        else:
            raise ValueError(name)

    @staticmethod
    def get_atom_feature_id(atom, name):
        """get atom features id"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return safe_index(CompoundKit.atom_vocab_dict[name], CompoundKit.get_atom_value(atom, name))

    @staticmethod
    def get_atom_feature_size(name):
        """get atom features size"""
        assert name in CompoundKit.atom_vocab_dict, "%s not found in atom_vocab_dict" % name
        return len(CompoundKit.atom_vocab_dict[name])

    ### bond

    @staticmethod
    def get_bond_value(bond, name):
        """get bond values"""
        if name == 'bond_dir':
            return bond.GetBondDir()
        elif name == 'bond_type':
            return bond.GetBondType()
        elif name == 'is_in_ring':
            return int(bond.IsInRing())
        elif name == 'is_conjugated':
            return int(bond.GetIsConjugated())
        elif name == 'bond_stereo':
            return bond.GetStereo()
        else:
            raise ValueError(name)

    @staticmethod
    def get_bond_feature_id(bond, name):
        """get bond features id"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return safe_index(CompoundKit.bond_vocab_dict[name], CompoundKit.get_bond_value(bond, name))

    @staticmethod
    def get_bond_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.bond_vocab_dict, "%s not found in bond_vocab_dict" % name
        return len(CompoundKit.bond_vocab_dict[name])

    ### plane
    @staticmethod
    def get_plane_feature_size(name):
        """get bond features size"""
        assert name in CompoundKit.plane_vocab_dict, "%s not found in plane_vocab_dict" % name
        return len(CompoundKit.plane_vocab_dict[name])

    ### fingerprint

    @staticmethod
    def get_morgan_fingerprint(mol, radius=2):
        """get morgan fingerprint"""
        nBits = CompoundKit.morgan_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]
    
    @staticmethod
    def get_morgan2048_fingerprint(mol, radius=2):
        """get morgan2048 fingerprint"""
        nBits = CompoundKit.morgan2048_fp_N
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return [int(b) for b in mfp.ToBitString()]

    @staticmethod
    def get_maccs_fingerprint(mol):
        """get maccs fingerprint"""
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return [int(b) for b in fp.ToBitString()]

    ### functional groups

    @staticmethod
    def get_daylight_functional_group_counts(mol):
        """get daylight functional group counts"""
        fg_counts = []
        for fg_mol in CompoundKit.day_light_fg_mo_list:
            sub_structs = Chem.Mol.GetSubstructMatches(mol, fg_mol, uniquify=True)
            fg_counts.append(len(sub_structs))
        return fg_counts

    @staticmethod
    def get_ring_size(mol):
        """return (N,6) list"""
        rings = mol.GetRingInfo()
        rings_info = []
        for r in rings.AtomRings():
            rings_info.append(r)
        ring_list = []
        for atom in mol.GetAtoms():
            atom_result = []
            for ringsize in range(3, 9):
                num_of_ring_at_ringsize = 0
                for r in rings_info:
                    if len(r) == ringsize and atom.GetIdx() in r:
                        num_of_ring_at_ringsize += 1
                if num_of_ring_at_ringsize > 8:
                    num_of_ring_at_ringsize = 9
                atom_result.append(num_of_ring_at_ringsize)
            
            ring_list.append(atom_result)
        return ring_list

    @staticmethod
    def atom_to_feat_vector(atom):
        """ tbd """
        atom_names = {
            "atomic_num": safe_index(CompoundKit.atom_vocab_dict["atomic_num"], atom.GetAtomicNum()),
            "chiral_tag": safe_index(CompoundKit.atom_vocab_dict["chiral_tag"], atom.GetChiralTag()),
            "degree": safe_index(CompoundKit.atom_vocab_dict["degree"], atom.GetTotalDegree()),
            "explicit_valence": safe_index(CompoundKit.atom_vocab_dict["explicit_valence"], atom.GetExplicitValence()),
            "formal_charge": safe_index(CompoundKit.atom_vocab_dict["formal_charge"], atom.GetFormalCharge()),
            "hybridization": safe_index(CompoundKit.atom_vocab_dict["hybridization"], atom.GetHybridization()),
            "implicit_valence": safe_index(CompoundKit.atom_vocab_dict["implicit_valence"], atom.GetImplicitValence()),
            "is_aromatic": safe_index(CompoundKit.atom_vocab_dict["is_aromatic"], int(atom.GetIsAromatic())),
            "total_numHs": safe_index(CompoundKit.atom_vocab_dict["total_numHs"], atom.GetTotalNumHs()),
            'num_radical_e': safe_index(CompoundKit.atom_vocab_dict['num_radical_e'], atom.GetNumRadicalElectrons()),
            'atom_is_in_ring': safe_index(CompoundKit.atom_vocab_dict['atom_is_in_ring'], int(atom.IsInRing())),
            'valence_out_shell': safe_index(CompoundKit.atom_vocab_dict['valence_out_shell'],
                                            CompoundKit.period_table.GetNOuterElecs(atom.GetAtomicNum())),
            'van_der_waals_radis': CompoundKit.period_table.GetRvdw(atom.GetAtomicNum()),
            'partial_charge': CompoundKit.check_partial_charge(atom),
            'mass': atom.GetMass(),
        }
        return atom_names

    @staticmethod
    def get_atom_names(mol):
        """get atom name list
        TODO: to be remove in the future
        """
        atom_features_dicts = []
        Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts.append(CompoundKit.atom_to_feat_vector(atom))

        ring_list = CompoundKit.get_ring_size(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features_dicts[i]['in_num_ring_with_size3'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size3'], ring_list[i][0])
            atom_features_dicts[i]['in_num_ring_with_size4'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size4'], ring_list[i][1])
            atom_features_dicts[i]['in_num_ring_with_size5'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size5'], ring_list[i][2])
            atom_features_dicts[i]['in_num_ring_with_size6'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size6'], ring_list[i][3])
            atom_features_dicts[i]['in_num_ring_with_size7'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size7'], ring_list[i][4])
            atom_features_dicts[i]['in_num_ring_with_size8'] = safe_index(
                    CompoundKit.atom_vocab_dict['in_num_ring_with_size8'], ring_list[i][5])

        return atom_features_dicts
        
    @staticmethod
    def check_partial_charge(atom):
        """tbd"""
        pc = atom.GetDoubleProp('_GasteigerCharge')
        if pc != pc:
            # unsupported atom, replace nan with 0
            pc = 0
        if pc == float('inf'):
            # max 4 for other atoms, set to 10 here if inf is get
            pc = 10
        return pc


class Compound3DKit(object):
    """the 3Dkit of Compound"""
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses

    @staticmethod
    def get_3d_atom_poses(mol, numConfs=None):
        """the atoms of mol will be changed in some cases."""
        try:
            mol_with_H = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(mol_with_H, numConfs=numConfs)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(mol_with_H)
            mol_wo_H = Chem.RemoveHs(mol_with_H)
            index = np.argmin([x[1] for x in res])
            conf = mol_wo_H.GetConformer(id=int(index))
        except:
            mol_wo_H = mol
            AllChem.Compute2DCoords(mol_wo_H)
            conf = mol_wo_H.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol_wo_H, conf)
        return mol_wo_H, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))
        bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""
        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)    # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0,], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles, bond_angle_dirs



# def new_smiles_to_graph_data(smiles, **kwargs):
#     """
#     Convert smiles to graph data.
#     """
#     mol = AllChem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     data = new_mol_to_graph_data(mol)
#     return data


# def new_mol_to_graph_data(mol):
#     """
#     mol_to_graph_data
#
#     Args:
#         atom_features: Atom features.
#         edge_features: Edge features.
#         morgan_fingerprint: Morgan fingerprint.
#         functional_groups: Functional groups.
#     """
#     if len(mol.GetAtoms()) == 0:
#         return None
#
#     atom_id_names = list(CompoundKit.atom_vocab_dict.keys()) + CompoundKit.atom_float_names
#     bond_id_names = list(CompoundKit.bond_vocab_dict.keys())
#
#     data = {}
#
#     ### atom features
#     data = {name: [] for name in atom_id_names}
#
#     raw_atom_feat_dicts = CompoundKit.get_atom_names(mol)
#     for atom_feat in raw_atom_feat_dicts:
#         for name in atom_id_names:
#             data[name].append(atom_feat[name])
#
#     ### bond and bond features
#     for name in bond_id_names:
#         data[name] = []
#     data['edges'] = []
#
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()
#         # i->j and j->i
#         data['edges'] += [(i, j), (j, i)]
#         for name in bond_id_names:
#             bond_feature_id = CompoundKit.get_bond_feature_id(bond, name)
#             data[name] += [bond_feature_id] * 2
#
#     #### self loop
#     N = len(data[atom_id_names[0]])
#     for i in range(N):
#         data['edges'] += [(i, i)]
#     for name in bond_id_names:
#         bond_feature_id = get_bond_feature_dims([name])[0] - 1   # self loop: value = len - 1
#         data[name] += [bond_feature_id] * N
#
#     ### make ndarray and check length
#     for name in list(CompoundKit.atom_vocab_dict.keys()):
#         data[name] = np.array(data[name], 'int64')
#     for name in CompoundKit.atom_float_names:
#         data[name] = np.array(data[name], 'float32')
#     for name in bond_id_names:
#         data[name] = np.array(data[name], 'int64')
#     data['edges'] = np.array(data['edges'], 'int64')
#
#     ### morgan fingerprint
#     data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
#     # data['morgan2048_fp'] = np.array(CompoundKit.get_morgan2048_fingerprint(mol), 'int64')
#     data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
#     data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
#     return data


def mol_to_basic_graph(mol):
    """
    mol_to_basic_graph

    Args:
        atom_features: Atom features.
        edge_features: Edge features.
        morgan_fingerprint: Morgan fingerprint.
        functional_groups: Functional groups.
    """
    if len(mol.GetAtoms()) == 0:
        return None

    atom_id_names = [
        "atomic_num", "chiral_tag", "degree", "explicit_valence", 
        "formal_charge", "hybridization", "implicit_valence", 
        "is_aromatic", "total_numHs",
    ]
    bond_id_names = [
        "bond_dir", "bond_type", "is_in_ring",
    ]
    
    data = {}
    for name in atom_id_names:
        data[name] = []
    data['mass'] = []
    for name in bond_id_names:
        data[name] = []
    data['edges'] = []

    ### atom features
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return None
        for name in atom_id_names:
            data[name].append(CompoundKit.get_atom_feature_id(atom, name) + 1)  # 0: OOV
        data['mass'].append(CompoundKit.get_atom_value(atom, 'mass') * 0.01)

    ### bond features
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # i->j and j->i
        data['edges'] += [(i, j), (j, i)]
        for name in bond_id_names:
            bond_feature_id = CompoundKit.get_bond_feature_id(bond, name) + 1   # 0: OOV
            data[name] += [bond_feature_id] * 2

    ### self loop (+2)
    N = len(data[atom_id_names[0]])
    for i in range(N):
        data['edges'] += [(i, i)]
    for name in bond_id_names:
        bond_feature_id = CompoundKit.get_bond_feature_size(name) + 2   # N + 2: self loop
        data[name] += [bond_feature_id] * N

    ### check whether edge exists
    if len(data['edges']) == 0: # mol has no bonds
        for name in bond_id_names:
            data[name] = np.zeros((0,), dtype="int64")
        data['edges'] = np.zeros((0, 2), dtype="int64")

    ### make ndarray and check length
    for name in atom_id_names:
        data[name] = np.array(data[name], 'int64')
    data['mass'] = np.array(data['mass'], 'float32')
    for name in bond_id_names:
        data[name] = np.array(data[name], 'int64')
    data['edges'] = np.array(data['edges'], 'int64')

    ### morgan fingerprint
    data['morgan_fp'] = np.array(CompoundKit.get_morgan_fingerprint(mol), 'int64')
    data['maccs_fp'] = np.array(CompoundKit.get_maccs_fingerprint(mol), 'int64')
    data['daylight_fg_counts'] = np.array(CompoundKit.get_daylight_functional_group_counts(mol), 'int64')
    return data


###############
def find_conjugated_systems(mol):
    """return conj edges"""
    conj_edge_index = []
    conjugated_bonds = [bond for bond in mol.GetBonds() if bond.GetIsConjugated()]
    visited = set()
    systems = []

    for bond in conjugated_bonds:
        if bond.GetIdx() in visited:
            continue
        current_system = set()
        stack = [bond]
        while stack:
            current_bond = stack.pop()
            if current_bond.GetIdx() in visited:
                continue
            visited.add(current_bond.GetIdx())
            a1 = current_bond.GetBeginAtomIdx()
            a2 = current_bond.GetEndAtomIdx()
            current_system.update([a1, a2])

            for atom in [a1, a2]:
                for nbr_bond in mol.GetAtomWithIdx(atom).GetBonds():
                    if nbr_bond.GetIsConjugated() and nbr_bond.GetIdx() not in visited:
                        stack.append(nbr_bond)
        if current_system:
            systems.append(list(current_system))


    for system in systems:
        system_atoms = set(system)
        for i in system_atoms:
            for j in system_atoms:
                if i != j and [i, j] not in conj_edge_index:
                    conj_edge_index.extend([[i, j], [j, i]])

    num_atoms = mol.GetNumAtoms()
    self_loops = [[i, i] for i in range(num_atoms)]
    all_edges = conj_edge_index + self_loops  # 合并共轭边和自环边

    return all_edges, systems  # 返回合并后的边列表和共轭系统

def get_bond_aromaticity_features(mol, conj_edge_index, conj_system):
    """process edges"""
    if conj_edge_index == [[0,0]]:
        return {
            'all_arom': np.array([0],'int64'),
            'hemi_arom': np.array([0],'int64'),
            'arom_conj': np.array([0],'int64'),
            'no_arom': np.array([0],'int64')
        }

    else:
        n_edges = len(conj_edge_index)
        features = {
            'all_arom': np.zeros((n_edges, 1)),
            'hemi_arom': np.zeros((n_edges, 1)),
            'arom_conj': np.zeros((n_edges, 1)),
            'no_arom': np.zeros((n_edges, 1))
        }


        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        def is_atom_aromatic(atom_idx):
            atom = mol.GetAtomWithIdx(atom_idx)
            if not atom.GetIsAromatic():
                return False
            return any(atom_idx in ring for ring in atom_rings)

        for idx, (i, j) in enumerate(conj_edge_index):
            if i == j:
                continue

            # wether aromatic
            atom_i_aromatic = is_atom_aromatic(i)
            atom_j_aromatic = is_atom_aromatic(j)

            if atom_i_aromatic and atom_j_aromatic:
                features['all_arom'][idx] = 1
                continue

            if atom_i_aromatic or atom_j_aromatic:
                features['hemi_arom'][idx] = 1
                continue

            def is_in_aromatic_conj_sys(atom_idx):
                for sys in conj_system:
                    if atom_idx in sys:
                        if any(is_atom_aromatic(a) for a in sys):
                            return True
                return False

            if is_in_aromatic_conj_sys(i) and is_in_aromatic_conj_sys(j):
                features['arom_conj'][idx] = 1
            else:
                features['no_arom'][idx] = 1

        return features

def add_spatial_feature(data, atom_poses):
    data['atom_pos'] = np.array(atom_poses, 'float32')
    data['bond_length'] = Compound3DKit.get_bond_lengths(data['edges'], data['atom_pos'])
    BondAngleGraph_edges, bond_angles, bond_angle_dirs = \
            Compound3DKit.get_superedge_angles(data['edges'], data['atom_pos'])
    data['BondAngleGraph_edges'] = BondAngleGraph_edges
    data['bond_angle'] = np.array(bond_angles, 'float32')
    data = add_diangle(data)
    return data

def add_conj_feature(data, mol):
    conj_edge, conj_system = find_conjugated_systems(mol)
    conj_edge_feature = get_bond_aromaticity_features(mol, conj_edge, conj_system)
    data['conj_edge'] = np.array(conj_edge, 'int64')
    if conj_edge == [[0, 0]]:
        data['all_arom'] = conj_edge_feature['all_arom']
        data['hemi_arom'] = conj_edge_feature['hemi_arom']
        data['arom_conj'] = conj_edge_feature['arom_conj']
        data['no_arom'] = conj_edge_feature['no_arom']
    else:
        data['all_arom'] = np.array(conj_edge_feature['hemi_arom'], 'int64').squeeze()
        data['hemi_arom'] = np.array(conj_edge_feature['hemi_arom'], 'int64').squeeze()
        data['arom_conj'] = np.array(conj_edge_feature['arom_conj'], 'int64').squeeze()
        data['no_arom'] = np.array(conj_edge_feature['no_arom'], 'int64').squeeze()
    return data


def mol_to_graph(mol):
    if len(mol.GetAtoms()) <= 400:
        mol, atom_poses = Compound3DKit.get_3d_atom_poses(mol, numConfs=10)
    else:
        atom_poses = Compound3DKit.get_2d_atom_poses(mol)
    if len(mol.GetAtoms()) == 0:
        return None
    data = mol_to_basic_graph(mol)
    data = add_conj_feature(data,mol)
    data = add_spatial_feature(data=data, atom_poses=atom_poses)
    return data



if __name__ == "__main__":
    smiles = "OCc1ccccc1CN"
    # smiles = r"[H]/[NH+]=C(\N)C1=CC(=O)/C(=C\C=c2ccc(=C(N)[NH3+])cc2)C=C1"
    mol = AllChem.MolFromSmiles(smiles)
    print(len(smiles))
    print(mol)
    data = mol_to_graph(mol)
    