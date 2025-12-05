import os
from os.path import join, exists
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from datasets.Load_dataset import LoadDataset

data_task = {'bace':['Class'],
             'bbbp':['p_np'],
             'clintox':['FDA_APPROVED', 'CT_TOX'],
             'sider':['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications'],
            'tox21':['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
             'hiv':['HIV_active'],
             'muv':['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
           'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
           'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'],
            'esol':['measured log solubility in mols per litre'],
            'lipophilicity':['exp'],
             'freesolv':['expt'],
             'qm7':['u0_atom'],
             'qm8':['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
            'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
            'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],
             'qm9':['homo', 'lumo', 'gap']
}





def get_default_toxcast_task_names(data_path):
    """Get that default toxcast task names and return the list of the input information"""
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    return list(input_df.columns)[1:]


def get_downstream_task_names(data_name,data_path):
    if data_name == 'toxcast':
        return get_default_toxcast_task_names(data_path)
    else:
        return data_task[data_name]



def get_esol_stat(data_path, task_names):
    """Return mean and std of labels"""
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }

def get_downstream_data_stat(data_name,data_path, task_names):
    """Return mean and std of labels"""
    # if data_name == 'am9':
    #     csv_file = join(data_path, 'raw/qm9.csv')
    #     input_df = pd.read_csv(csv_file, sep=',')
    #     labels = input_df[task_names].values
    #     return {
    #         'mean': np.mean(labels, 0),
    #         'std': np.std(labels, 0),
    #         'N': len(labels),
    #     }
    # else:
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }

def load_downstream_dataset(data_name ,data_path, task_names=None):
    if task_names is None:
        task_names = get_downstream_task_names(data_name=data_name, data_path=data_path)
    raw_path = join(data_path, 'raw')
    csv_file = os.listdir(raw_path)[0]
    input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
    if data_name == 'bace':
        smiles_list = input_df['mol']
    else:
        smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if not m is None else None for m in rdkit_mol_objs_list]
    smiles_list = [AllChem.MolToSmiles(m) if not m is None else None for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df[task_names]
    # labels = labels.replace(0, -1)
    if data_name == 'tox21':
        labels = labels.fillna(0.5)
    elif data_name == 'toxcast':
        labels = labels.fillna(0.5)
    elif data_name == 'hiv':
        labels = labels.fillna(0.5)
    elif data_name == 'muv':
        labels = labels.fillna(0.5)
    data_list = []
    for i in range(len(smiles_list)):
        if smiles_list[i] is None:
            continue
        data = {
            'smiles': smiles_list[i],
            'label': labels.values[i],
        }
        data_list.append(data)
    dataset = LoadDataset(data_list)
    return dataset
# def get_lipophilicity_stat(data_path, task_names):
#     """Return mean and std of labels"""
#     raw_path = join(data_path, 'raw')
#     csv_file = os.listdir(raw_path)[0]
#     input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
#     labels = input_df[task_names].values
#     return {
#         'mean': np.mean(labels, 0),
#         'std': np.std(labels, 0),
#         'N': len(labels),
#     }
# def get_qm7_stat(data_path, task_names):
#     """Return mean and std of labels"""
#     raw_path = join(data_path, 'raw')
#     csv_file = os.listdir(raw_path)[0]
#     input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
#     labels = input_df[task_names].values
#     return {
#         'mean': np.mean(labels, 0),
#         'std': np.std(labels, 0),
#         'N': len(labels),
#     }
# def get_qm8_stat(data_path, task_names):
#     """Return mean and std of labels"""
#     raw_path = join(data_path, 'raw')
#     csv_file = os.listdir(raw_path)[0]
#     input_df = pd.read_csv(join(raw_path, csv_file), sep=',')
#     labels = input_df[task_names].values
#     return {
#         'mean': np.mean(labels, 0),
#         'std': np.std(labels, 0),
#         'N': len(labels),
#     }
def get_qm9_stat(data_path, task_names):
    """Return mean and std of labels"""
    csv_file = join(data_path, 'raw/qm9.csv')
    input_df = pd.read_csv(csv_file, sep=',')
    labels = input_df[task_names].values
    return {
        'mean': np.mean(labels, 0),
        'std': np.std(labels, 0),
        'N': len(labels),
    }