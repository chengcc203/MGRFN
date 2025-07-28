import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import sys
import torch.nn.functional as F
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
from dataset_pro import Chiral
from torch_geometric.data import DataLoader
from method.MGRFN import MGRFN
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

def val(model, data_loader, loss_func):
    model.eval()
    val_loss = 0
    val_acc = 0
    all_features = []
    all_labels = []

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out, mol_feature = model(batch_data)
            loss = loss_func(out, batch_data.y.unsqueeze(1))
            val_loss += loss.detach().cpu().item()
            predicted = torch.round(torch.sigmoid(out))
            val_acc += ((predicted == batch_data.y.unsqueeze(1)).sum().double()) / len(batch_data)
            all_features.append(mol_feature.cpu().numpy())
            all_labels.append(batch_data.y.cpu().numpy())

    return val_loss / (step + 1), val_acc / (step + 1), all_features, all_labels

def test(model, data_loader, loss_func):

    all_features = []
    all_labels = []

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out, mol_feature = model(batch_data)
            all_features.append(mol_feature.cpu().numpy())
            all_labels.append(batch_data.y.cpu().numpy())
    molecular_features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return molecular_features, labels

if __name__ == '__main__':
    dataset = Chiral()
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=2965, valid_size=370, test_size=370,
                                      seed=120)
    train_dataset = dataset[split_idx['train']]
    valid_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]
    model = MGRFN(energy_and_force=True, cutoff=5.0, num_layers=4,
                hidden_channels=128, out_channels=1, int_emb_size=64,
                basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                num_spherical=3, num_radial=6, envelope_exponent=5,
                num_before_skip=1, num_after_skip=2, num_output_layers=3, use_node_features=True
                ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')
    optimizer = Adam(model.parameters(), lr=0.0002, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    min_loss = float('inf')
    save_dir = 'checkpoint/chiral/'

    for epoch in range(1, 65):
        model.train()
        loss_accum = 0
        train_acc = 0
        train_features = []
        train_labels = []
        print("\n=====Epoch {}".format(epoch), flush=True)

        print('\nTraining...', flush=True)
        for step, batch_data in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out, train_feature = model(batch_data)
            loss = F.binary_cross_entropy_with_logits(out, batch_data.y.unsqueeze(1))
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().cpu().item()
            predicted = torch.round(torch.sigmoid(out))
            train_acc += ((predicted == batch_data.y.unsqueeze(1)).sum().double()) / len(batch_data)
            # train_features.append((train_feature.cpu().numpy())
            # train_labels.append()

        total_loss = loss_accum / (step + 1)
        train_acc = train_acc / (step + 1)

        print('\n\nEvaluating...', flush=True)

        valid_loss, valid_acc, valid_features, val_labels = val(model, valid_loader, F.binary_cross_entropy_with_logits)

        print('\n\nTesting...', flush=True)
        test_loss, test_acc, test_features, test_labels = val(model, test_loader, F.binary_cross_entropy_with_logits)

        print({'Train': {total_loss, round(train_acc.item(), 4)}, 'Validation': round(valid_acc.item(), 4), 'Test': round(test_acc.item(), 4)})

        if min_loss > test_loss:
            min_loss = test_loss
            if save_dir != '':
                molecular_features = []
                labels = []
                molecular_features.append(np.concatenate(test_features, axis=0))
                molecular_features.append(np.concatenate(valid_features, axis=0))
                labels.append(np.concatenate(val_labels, axis=0))
                labels.append(np.concatenate(test_labels, axis=0))
                np.savez(
                    "/dataset/chiral/representation/chiral_representations.npz",
                    features=np.concatenate(molecular_features, axis=0),
                    labels=np.concatenate(labels, axis=0),
                )
                print('Saving checkpoint...')
                torch.save(model.state_dict(), os.path.join(save_dir, 'Chiral_checkpoint.pt'))

        scheduler.step()

        file1 = open('mae/chiral/loss_acc.txt', 'a')
        print(total_loss, valid_acc, test_acc, file=file1)
        file1.close()


    print(f'Min loss so far: {min_loss}')

