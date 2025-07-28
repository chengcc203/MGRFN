import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from rdkit.Chem import AllChem
from sklearn.utils import shuffle
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader
import pandas as pd
from rdkit import Chem
from openbabel import pybel


def get_atom_features(atom):
    atom_type = atom.GetHybridization()
    chiral_tag = atom.GetChiralTag()
    atom_feats = [
        int(atom.GetIsAromatic() == True), int(atom.GetIsAromatic() == False),
        int(atom_type == Chem.rdchem.HybridizationType.SP),
        int(atom_type == Chem.rdchem.HybridizationType.SP2),
        int(atom_type == Chem.rdchem.HybridizationType.SP3),
        int(atom_type == Chem.rdchem.HybridizationType.SP3D),
        int(atom_type == Chem.rdchem.HybridizationType.UNSPECIFIED),
        # int(chiral_tag != ChiralType.CHI_UNSPECIFIED),
        # int(chiral_tag == ChiralType.CHI_TETRAHEDRAL_CW),  # 四面体顺时针
        # int(chiral_tag == ChiralType.CHI_TETRAHEDRAL_CCW),  # 四面体逆时针
        # int(chiral_tag == ChiralType.CHI_OTHER)
    ]
    return np.array(atom_feats)


def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def get_3d_coordinates(smiles):
    mol = Chem.MolFromSmiles(smiles)

    try:
        # 生成单个三维构象
        AllChem.EmbedMolecule(mol)
        # 优化分子结构
        AllChem.MMFFOptimizeMolecule(mol)

        return np.array(mol.GetConformer().GetPositions())

    except Exception as e:
        print(f"RDKit failed to generate coordinates: {e}")
        t_mol = Chem.MolFromSmiles(smiles)
        mol_block = Chem.MolToMolBlock(t_mol)

        # 使用 Pybel 创建分子对象
        pybel_mol = pybel.readstring('mol', mol_block)
        pybel_mol.localopt()  # 优化分子

        # 提取 3D 坐标
        coords = []
        for atom in pybel_mol:
            if atom.atomicnum != 1:  # 过滤氢原子
                coords.append(atom.coords)
        return np.array(coords)


class Chiral(InMemoryDataset):

    def __init__(self, root='dataset/', transform=None, pre_transform=None, pre_filter=None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'chiral')
        print(self.folder)
        self.afeat = []
        self.bfeat = []
        super(Chiral, self).__init__(self.folder, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'chiralPairs.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        path = "/dataset/chiral/raw/chiralPairs.csv"
        df = pd.read_csv(path)
        data_list = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            smiles = row['smiles']
            sdf_name = "/dataset/chiral/raw/pos/" + str(i) + ".sdf"
            if not os.path.exists(sdf_name):
                mol = get_3d_coordinates(smiles)
            # 解析sdf内容
            mol = Chem.MolFromMolFile(sdf_name)
            a_feature, b_feature = [], []
            for atom in mol.GetAtoms():
                a_feature.append(get_atom_features(atom))
            for bond in mol.GetBonds():
                b_feature.append(get_bond_features(bond))
            Z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.int64)
            R = torch.tensor(np.array(mol.GetConformer().GetPositions()), dtype=torch.float32)
            Y = torch.tensor(row['class'], dtype=torch.float)
            a_fea = torch.tensor(a_feature)
            b_fea = torch.tensor(b_feature)
            data = Data(pos=R, z=Z, y=Y, afe=a_fea, bfe=b_fea)
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        self.data = data
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, test_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        if seed == False:
            ids = range(data_size)
        train_idx, val_idx, test_idx = torch.tensor(ids[ : train_size]), torch.tensor(ids[train_size : train_size + valid_size]), torch.tensor(ids[train_size + valid_size : train_size + valid_size + test_size])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict