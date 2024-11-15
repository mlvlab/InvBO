import pandas as pd
import torch
import math
import selfies as sf
import torch.nn.functional as F
from invbo.utils.mol_utils.selfies_vae.data import collate_fn
from tqdm import tqdm

def load_molecule_train_data(
    task_id,
    path_to_vae_statedict,
    num_initialization_points=100,
): 
    df_origin = pd.read_csv("../invbo/utils/mol_utils/guacamol_data/guacamol_train_data_first_100.csv")
    df = df_origin[0:num_initialization_points]
    train_x_smiles = df['smile'].values.tolist()
    train_x_selfies = df['selfie'].values.tolist() 
    train_y = torch.from_numpy(df[task_id].values).float() 
    train_y = train_y.unsqueeze(-1)
    train_z = load_train_z(
        num_initialization_points=num_initialization_points,
        path_to_vae_statedict=path_to_vae_statedict,
    ) 

    return train_x_smiles, train_x_selfies, train_z, train_y

def load_train_z(
    num_initialization_points,
    path_to_vae_statedict,
):
    state_dict_file_type = path_to_vae_statedict.split('.')[-1]
    path_to_init_train_zs = path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
    try:
        zs = pd.read_csv(path_to_init_train_zs, header=None).values
        assert len(zs) >= num_initialization_points
        zs = zs[0:num_initialization_points]
        zs = torch.from_numpy(zs).float()
    except: 
        zs = None 

    return zs

def compute_train_zs(
    mol_objective,
    init_train_x,
    bsz=100,
):
    n_batches = math.ceil(len(init_train_x)/bsz)
    init_z = torch.zeros([0]).cuda()
    mol_objective.vae.eval() 

    with torch.no_grad():
        for i in range(n_batches):
            xs_batch = init_train_x[i*bsz:(i+1)*bsz] 
            X_list = []
            for smile in xs_batch:
                try:
                    selfie = mol_objective.smiles_to_selfies[smile]
                except:
                    selfie = sf.encoder(smile)
                    mol_objective.smiles_to_selfies[smile] = selfie
                tokenized_selfie = mol_objective.dataobj.tokenize_selfies([selfie])[0]
                encoded_selfie = mol_objective.dataobj.encode(tokenized_selfie).unsqueeze(0)
                X_list.append(encoded_selfie)
            tokens = collate_fn(X_list)    
            z, _ = mol_objective.vae.encode(tokens.cuda())
            init_z = torch.cat((init_z, z), dim=0)
    
    init_z.requires_grad_()

    print('Start Initial data inversion')
    
    final_z = torch.zeros_like(init_z)
    finish_idx = []
    for i in tqdm(range(n_batches)):
        optimizer = torch.optim.Adam([
            {'params': init_z, 'lr': 1e-1},
        ])
        config = init_train_x[i*bsz:(i+1)*bsz]
        input_z = init_z[i*bsz:(i+1)*bsz].cuda() 

        X_list = []
        for smile in config:
            try:
                selfie = mol_objective.smiles_to_selfies[smile]
            except:
                selfie = sf.encoder(smile)
                mol_objective.smiles_to_selfies[smile] = selfie
            tokenized_selfie = mol_objective.dataobj.tokenize_selfies([selfie])[0]
            encoded_selfie = mol_objective.dataobj.encode(tokenized_selfie).unsqueeze(0)
            X_list.append(encoded_selfie)
        X = collate_fn(X_list)

        for e in range(1000): 
            optimizer.zero_grad()
            mol_objective.vae.zero_grad()

            logits = mol_objective.vae.decode(input_z, X.cuda())
            loss = F.cross_entropy(logits.permute(0, 2, 1), X.cuda())
            
            mean_acc = (logits.argmax(dim=-1) == X.cuda()).float().mean(-1)
            stop_idx = torch.where(mean_acc == 1)[0]
            
            if len(stop_idx) != 0:
                remove_idx = []
                for idx in stop_idx:
                    if (idx.item() + i*bsz) not in finish_idx:
                        finish_idx.append(idx.item() + i*bsz)
                        remove_idx.append(idx.item() + i*bsz)
                final_z[remove_idx] = init_z[remove_idx].detach()
            
            if (1 - (logits.argmax(dim=-1) == X.cuda()).float().mean()) < 1e-9:
                break

            loss.backward()
            optimizer.step()
    
    non_one_idx = torch.tensor([i for i in [*range(len(init_z))] if i not in finish_idx])
    if len(non_one_idx) != 0:
        final_z[non_one_idx] = init_z[non_one_idx].detach()
    final_z = final_z.reshape(final_z.shape[0], -1)
    
    state_dict_file_type = mol_objective.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
    path_to_init_train_zs = mol_objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", f'-train-zs.csv')
    zs_arr = final_z.cpu().detach().numpy()
    pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None) 
    return final_z.cpu()