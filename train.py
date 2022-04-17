import os
import sys
import json
import wandb
import torch
import pickle
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm
from torch.utils.data import DataLoader
from stgcn.model import STGCN
from stgcn.utils import normalize_matrix, process_adj_mat, TimeSpaceDataset
from pytorch_forecasting.metrics import MAE, MAPE

def train(config_path):
    with open(config_path, 'r') as config_json:
        config = json.load(config_json)

    data_config = config['data_config']
    training_config = config['training_config']
    model_config = config['model_config']
    optimizer_config = config['optimizer_config']
    scheduler_config = config['scheduler_config']

    with open(data_config['time_space_matrix_path'], 'rb') as file:
        time_space_matrix = pickle.load(file)

    train_time_space_matrix = time_space_matrix[:-data_config['val_timesteps_split'], :]
    val_time_space_matrix = time_space_matrix[-data_config['val_timesteps_split']:, :]

    z_score_mean = train_time_space_matrix.mean()
    z_score_sigma = np.sqrt(train_time_space_matrix.var())

    train_time_space_matrix = (train_time_space_matrix-z_score_mean)/z_score_sigma
    val_time_space_matrix = (val_time_space_matrix-z_score_mean)/z_score_sigma

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    with open(data_config['adj_matrix_path'], 'rb') as pickle_file:
        adj_mat = pickle.load(pickle_file)

    A_hat = process_adj_mat(adj_mat, eps=data_config['preprocess_eps'], sigma_sq=data_config['preprocess_sigma_sq'])
    A_hat = torch.from_numpy(normalize_matrix(A_hat)).to(device, dtype=torch.float)

    model = STGCN(A_hat, 1, model_config['in_timesteps'], model_config['out_timesteps']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_config['gamma'])
    loss_criterion = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()

    train_dataset = TimeSpaceDataset(
        time_space_matrix=train_time_space_matrix, 
        in_timesteps=model_config['in_timesteps'], 
        out_timesteps=model_config['out_timesteps']
        )
    val_dataset = TimeSpaceDataset(
        time_space_matrix=val_time_space_matrix, 
        in_timesteps=model_config['in_timesteps'], 
        out_timesteps=model_config['out_timesteps']
        )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        pin_memory=True, 
        num_workers=training_config['num_workers']
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size'], 
        pin_memory=True, 
        num_workers=training_config['num_workers']
        )

    out_dir = training_config['out_dir']
    training_stats = {
        'epoch': [],
        'avg_train_mse_loss': [],
        'avg_train_mae_loss': [],
        'avg_train_mape_loss': [],
        'avg_val_mse_loss': [],
        'avg_val_mae_loss': [],
        'avg_val_mape_loss': []
    }

    with tqdm(total=training_config['num_epochs']) as pbar:
        for epoch in range(training_config['num_epochs']):
            model.train()
            total_train_mse_loss = 0
            total_train_mae_loss = 0
            total_train_mape_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                features, target = batch
                features = (features-z_score_mean)/z_score_sigma

                features = torch.unsqueeze(features, 3).to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)
                
                predict = model(features)
                train_mse_loss = loss_criterion(predict, target)

                with torch.no_grad():
                    total_train_mse_loss += train_mse_loss.cpu().item()
                    total_train_mae_loss += torch.mean(MAE().loss(predict.squeeze().unsqueeze(-1), target.squeeze())).cpu().item()
                    total_train_mape_loss += torch.mean(MAPE().loss(predict.squeeze().unsqueeze(-1), target.squeeze())).cpu().item()

                train_mse_loss.backward()
                optimizer.step()

            if (epoch+1)%scheduler_config['decay_epoch']==0:
                scheduler.step()

            training_stats['avg_train_mse_loss'].append(total_train_mse_loss/len(train_dataset))
            training_stats['avg_train_mae_loss'].append(total_train_mae_loss/len(train_dataset))
            training_stats['avg_train_mape_loss'].append(total_train_mape_loss/len(train_dataset))

            model.eval()
            total_val_mse_loss = 0
            total_val_mae_loss = 0
            total_val_mape_loss = 0
            for batch in val_dataloader:
                with torch.no_grad():
                    features, target = batch

                    features = torch.unsqueeze(features, 3).to(device, dtype=torch.float)
                    target = target.to(device, dtype=torch.float)

                    predict = model(features)
                    val_mse_loss = loss_criterion(predict, target)

                    total_val_mse_loss += val_mse_loss.cpu().item()
                    total_val_mae_loss += torch.mean(MAE().loss(predict.squeeze().unsqueeze(-1), target.squeeze())).cpu().item()
                    total_val_mape_loss += torch.mean(MAPE().loss(predict.squeeze().unsqueeze(-1), target.squeeze())).cpu().item()

            training_stats['avg_val_mse_loss'].append(total_val_mse_loss/len(val_dataset))
            training_stats['avg_val_mae_loss'].append(total_val_mae_loss/len(val_dataset))
            training_stats['avg_val_mape_loss'].append(total_val_mape_loss/len(val_dataset))
            
            training_stats['epoch'].append(epoch+1)

            pbar.set_description(
                f'epoch: {epoch+1}, '\
                f'avg_train_loss: {round(total_train_mse_loss/len(train_dataset), 6)}, ' \
                f'avg_val_loss: {round(total_val_mse_loss/len(val_dataset), 6)}, '\
                f'lr: {scheduler.get_last_lr()[0]}'
                )
            pbar.update(1)

            pd.DataFrame(training_stats).to_csv(f'{out_dir}/training_results.csv', index=False)

            wandb.log(
                {key: val[-1] for key, val in training_stats.items()}
            )
    
if __name__ == '__main__':
    load_dotenv()
    WANDB_PROJECT_KEY = os.getenv('WANDB_PROJECT_KEY', '')
    wandb.login(key=WANDB_PROJECT_KEY)
    wandb.init(project="tugas-akhir", entity="mikhaelbelmiro")
    config_path = sys.argv[1]
    train(config_path)