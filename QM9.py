import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
from dataset_pro import QM9
from method.MGRFN import MGRFN
from method.run_qm9 import run_qm9
from dig.threedgraph.evaluation import ThreeDEvaluator


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print(device)
dataset = QM9(root='dataset/')
target = 'lumo'
dataset.data.y = dataset.data[target]
print(len(dataset))
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=600, valid_size=100, test_size=100, seed=120)

print([split_idx['train']])
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

type=2
num1=1000
idxs = np.random.randint(42, num1, size=1000)
idxs=split_idx['train'][idxs]
if type==0:
    print(dataset.data.y[idxs])
    dataset.data.y[idxs]=1.05*dataset.data.y[idxs]
    print(dataset.data.y[idxs])
if type==1:
    print(dataset.data.pos)
    dataset.data.pos[0] = 0*dataset.data.pos[0]
    #dataset.data.pos[1]=0*dataset.data.pos[1]
    print(dataset.data.pos)

model = MGRFN(energy_and_force=True, cutoff=5.0, num_layers=4,
        hidden_channels=128, out_channels=1, int_emb_size=64,
        basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        num_spherical=3, num_radial=6, envelope_exponent=5,
        num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
        )

loss_func = torch.nn.L1Loss()
evaluation = ThreeDEvaluator()
run3d = run_qm9()
run3d.run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, evaluation, epochs=200, batch_size=32, vt_batch_size=64, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=15, tar=target)