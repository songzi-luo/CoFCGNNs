from datasets.Load_dataset import LoadDataset,save_data_list_to_npz
import pgl
import os
import json
import time
import multiprocessing
from multiprocessing import Pool
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from utils.compound_tools import *
from featurizers.feature_abstracter import PredTransformFn, PredCollateFn
from rdkit import Chem
# from rdkit.Chem import RDLogger
import glob
import rdkit.Chem as Chem



class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)


class QuantumFeatureExtractor:
    def __init__(self, json_dir: str, output_dir: str, num_workers: int = None):
        self.json_dir = json_dir
        self.output_dir = output_dir
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)

        self.file_counter = 0
        self.processed_counter = 0
        self.valid_mol_counter = 0
        self.num_fixed_smile = 0
        self.num_fix_failed = 0

        os.makedirs(output_dir, exist_ok=True)

    def get_json_files(self) -> List[str]:
        return [os.path.join(dp, f)
                for dp, dn, fn in os.walk(self.json_dir)
                for f in fn if f.endswith('.json')]

    def sanitize_valence_errors(self, mol: Chem.Mol, original_smiles: str, json_path: str) -> Tuple[
        Optional[Chem.Mol], Optional[str]]:
        """fix exp_valence"""
        if mol is None:
            return None, None

        try:
            # standard sanitization
            Chem.SanitizeMol(mol)
            return mol, original_smiles
        except ValueError as e:
            if "Explicit valence" not in str(e):
                return None, None


            rw_mol = Chem.RWMol(mol)
            modified = False
            ###new
            for atom in rw_mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                exp_valence = atom.GetExplicitValence()

                # process error element B
                if atomic_num == 5 and exp_valence > 3:
                    atom.SetFormalCharge(-1 if exp_valence == 4 else 0)
                    modified = True
                # process error element N
                elif atomic_num == 7 and exp_valence > 3:
                    atom.SetFormalCharge(+1)
                    modified = True
            if modified:
                try:
                    Chem.SanitizeMol(rw_mol)
                    fixed_smiles = Chem.MolToSmiles(rw_mol, isomericSmiles=True, canonical=False)
                    print('orig smile',original_smiles)
                    print('fix smile',fixed_smiles)
                    self.num_fixed_smile += 1
                    return rw_mol, fixed_smiles
                except Exception as fix_e:
                    print(f"Fix failed: {str(fix_e)}")
                    print("Original SMILES:", original_smiles)
                    print('name:', json_path)
                    self.num_fix_failed += 1
                    return None, None
            return None, None


    def _validate_smiles(self, smiles: str, json_path: str) -> Tuple[bool, Optional[str]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return True, smiles
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False, None
        mol, fixed_smiles = self.sanitize_valence_errors(mol, smiles, json_path)
        if mol is None:
            return False, None
        return True, fixed_smiles if fixed_smiles else smiles

    def extract_features_from_file(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)

        smiles = data.get('smiles', '')
        is_valid, processed_smiles = self._validate_smiles(smiles, json_path)
        final_smiles = processed_smiles if is_valid and processed_smiles else smiles

        mulliken_charges = [
            charge for charge, atomic_num in zip(
                data.get('properties', {}).get('partial charges', {}).get('mulliken', []),
                data.get('atoms', {}).get('elements', {}).get('number', [])
            ) if atomic_num != 1
        ]

        return {
            'smiles': final_smiles,
            'original_smiles': smiles,
            'mulliken_charges': mulliken_charges,
            'homo': (
                data.get('properties', {}).get('energy', {}).get('alpha', {}).get('homo', 0)
            ),
            'lumo': (
                data.get('properties', {}).get('energy', {}).get('alpha', {}).get('lumo', 0)
            ),
            'dipole':
                data.get('properties', {}).get('total dipole moment', [0, 0, 0]),
            'valid_mol': is_valid,
            'was_fixed': processed_smiles is not None and processed_smiles != smiles,
            'source_file': os.path.basename(json_path)
        }

    def process_single_file_wrapper(self, json_path: str) -> Optional[Dict[str, Any]]:
        """process single file"""
        try:
            features = self.extract_features_from_file(json_path)

            if not features.get('valid_mol', False):
                return None


            try:
                transform_fn = PredTransformFn(pretrain_tasks=None, mask_ratio=None)
                fea = transform_fn(features['smiles'])

                num_nodes = len(fea['atomic_num'])
                fea['homo'] = np.array([features['homo']],dtype='float32')
                fea['lumo'] = np.array([features['lumo']],dtype='float32')
                fea['dipole'] = np.array([features['dipole']],dtype='float32')
                fea['mulliken_charges'] = np.array(features['mulliken_charges'],dtype='float32')
                if num_nodes == len(fea['mulliken_charges']):
                    if len(fea['BondAngleGraph_edges']) == 0:
                        return None
                    return fea
                elif len(fea['atomic_num']) != len(fea['mulliken_charges']):
                    print('notequal skip')

            except Exception as e:
                print(f"Geometric transform failed for {features['source_file']}: {e}")
                return None

        except Exception as e:
            print(f"Wrapper error for {json_path}: {e}")
            return None

    def process_with_multiprocessing(self, chunk_size: int = 1000):
        """muti_process"""
        json_files = self.get_json_files()
        file_counter = 0

        print(f"Found {len(json_files)} JSON files")
        print(f"Using {self.num_workers} processes")
        print(f"Chunk size: {chunk_size} molecules per file")

        start_time = time.time()
        chunk_buffer = []


        with Pool(processes=self.num_workers) as pool:

            results = pool.imap_unordered(self.process_single_file_wrapper, json_files, chunksize=100)
            for i, result in enumerate(results):
                if result is not None:
                    chunk_buffer.append(result)
                    self.valid_mol_counter += 1


                if i % 1000 == 0 and i > 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {i}/{len(json_files)} files, "
                          f"valid molecules: {self.valid_mol_counter}, "
                          f"elapsed: {elapsed:.2f}s")
                # save
                if len(chunk_buffer) >= chunk_size:
                    output_file = os.path.join(args.output_dir, f"part-{file_counter:06d}.npz")
                    save_data_list_to_npz(chunk_buffer, output_file)

                    print(f"Saved chunk {file_counter} with {len(chunk_buffer)} molecules to {output_file}")

                    chunk_buffer = []
                    file_counter += 1
                    self.file_counter += 1

                self.processed_counter += 1

            if chunk_buffer:
                output_file = os.path.join(args.output_dir, f"part-{file_counter:06d}.npz")
                save_data_list_to_npz(chunk_buffer, output_file)

                print(f"Saved chunk {file_counter} with {len(chunk_buffer)} molecules to {output_file}")
                self.file_counter += 1


        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"Processing completed!")
        print(f"Total files processed: {self.processed_counter}")
        print(f"Valid molecules extracted: {self.valid_mol_counter}")
        print(f"Fixed SMILES: {self.num_fixed_smile}")
        print(f"Failed fixes: {self.num_fix_failed}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per file: {total_time / len(json_files):.4f} seconds")
        print(f"Output chunks: {self.file_counter} files")
        print(f"{'=' * 60}")

    def save_chunk(self, chunk_data: List[Dict[str, Any]]):

        chunk_num = self.file_counter
        output_file = os.path.join(self.output_dir, f'molecule_chunk_{chunk_num:06d}.json')

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False, cls=NumpyArrayEncoder)

            print(f"Saved chunk {chunk_num} with {len(chunk_data)} molecules to {output_file}")
            self.file_counter += 1

        except Exception as e:
            print(f"Error saving chunk {chunk_num}: {e}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=rf'../../pretrain_data/json1')
    parser.add_argument('--output_dir', type=str, default='../../pretrain_catch_data')
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--test_size', type=int, default=None)

    args = parser.parse_args()

    total_start_time = time.time()

    extractor = QuantumFeatureExtractor(
        json_dir=args.data_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers
    )

    if args.test_size:
        test_files = extractor.get_json_files()[:args.test_size]
        print(f"Testing speed with first {args.test_size} files...")
        print(f"Total files available: {len(extractor.get_json_files())}")

        test_start_time = time.time()
        # 这里可以添加测试逻辑
        extractor.process_with_multiprocessing(chunk_size=args.chunk_size)
        test_time = time.time() - test_start_time

        print(f"First {args.test_size} files processed in {test_time:.2f} seconds")
        print(f"Average time per file: {test_time / args.test_size:.4f} seconds")
    else:
        print('all')
        extractor.process_with_multiprocessing(chunk_size=args.chunk_size)

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.2f} seconds")