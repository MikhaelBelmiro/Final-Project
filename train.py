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

from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from stgcn.model import STGCN
from stgcn.utils import normalize_matrix, process_adj_mat, TimeSpaceDataset, MSE, MAE, MAPE

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

    scaler = StandardScaler()
    train_time_space_matrix = scaler.fit_transform(train_time_space_matrix)
    val_time_space_matrix = scaler.fit_transform(val_time_space_matrix)

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    with open(data_config['adj_matrix_path'], 'rb') as pickle_file:
        adj_mat = pickle.load(pickle_file)

    A_hat = process_adj_mat(adj_mat, eps=data_config['preprocess_eps'], sigma_sq=data_config['preprocess_sigma_sq'])
    A_hat = torch.from_numpy(normalize_matrix(A_hat)).to(device, dtype=torch.float)

    model = STGCN(A_hat, 1, model_config['in_timesteps'], model_config['out_timesteps']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, scheduler_config['gamma'])
    loss_criterion = torch.nn.MSELoss()

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
        num_workers=training_config['num_workers'],
        shuffle=True
        )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size'], 
        pin_memory=True, 
        num_workers=training_config['num_workers'],
        shuffle=True
        )

    out_dir = training_config['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    training_stats = {
        'epoch': [],
        'avg_train_mse_loss': [],
        'avg_train_mae_loss': [],
        'avg_train_mape_loss': [],
        'avg_val_mse_loss': [],
        'avg_val_mae_loss': [],
        'avg_val_mape_loss': [],
        'early_stopping_patience': []
    }
    with tqdm(total=training_config['num_epochs']) as pbar:
        early_stopping_patience = training_config['early_stopping_patience']
        early_stopping_delta = training_config['early_stopping_delta']

        for epoch in range(training_config['num_epochs']):
            model.train()
            total_train_mse_loss = 0
            total_train_mae_loss = 0
            total_train_mape_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                features, target = batch

                features = torch.unsqueeze(features, 3).to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)
                
                predict = model(features)
                loss = loss_criterion(predict, target)

                with torch.no_grad():
                    predict = scaler.inverse_transform(predict.squeeze().cpu().numpy())
                    target = scaler.inverse_transform(target.squeeze().cpu().numpy())

                    total_train_mse_loss += MSE(target, predict)
                    total_train_mae_loss += MAE(target, predict)
                    total_train_mape_loss += MAPE(target, predict)

                loss.backward()
                optimizer.step()

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

                    predict = scaler.inverse_transform(predict.squeeze().cpu().numpy())
                    target = scaler.inverse_transform(target.squeeze().cpu().numpy())

                    total_val_mse_loss += MSE(target, predict)
                    total_val_mae_loss += MAE(target, predict)
                    total_val_mape_loss += MAPE(target, predict)

            training_stats['avg_val_mse_loss'].append(total_val_mse_loss/len(val_dataset))
            training_stats['avg_val_mae_loss'].append(total_val_mae_loss/len(val_dataset))
            training_stats['avg_val_mape_loss'].append(total_val_mape_loss/len(val_dataset))
                        
            if (epoch+1)%scheduler_config['decay_epoch']==0:
                scheduler.step()

            if len(training_stats['avg_val_mape_loss']) > 3:
                if abs(training_stats['avg_val_mape_loss'][-1]-training_stats['avg_val_mape_loss'][-2])<early_stopping_delta:
                    early_stopping_patience -= 1
                    if early_stopping_patience == 0:
                        break
                else:
                    early_stopping_patience = training_config['early_stopping_patience']

            training_stats['epoch'].append(epoch+1)
            training_stats['early_stopping_patience'].append(early_stopping_patience)

            pbar.set_description(
                f'epoch: {epoch+1}, '\
                f'avg_train_loss: {round(total_train_mse_loss/len(train_dataset), 3)}, ' \
                f'avg_val_loss: {round(total_val_mse_loss/len(val_dataset), 3)}, '\
                f'patience: {early_stopping_patience}'
                )
            pbar.update(1)

            pd.DataFrame(training_stats).to_csv(f'{out_dir}/training_results.csv', index=False)

            rank = sorted(training_stats['avg_val_mape_loss']).index(training_stats['avg_val_mape_loss'][-1]) + 1
            if rank < training_config['save_n_best']:
                torch.save(model, f'{out_dir}/model-rank{rank}.pth')

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