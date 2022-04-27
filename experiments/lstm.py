import sys
sys.path.append('.')

import os
import time
import json
import wandb
import torch
import pickle

import pandas as pd

from tqdm import tqdm
from models import LSTMModel
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from stgcn.utils import TimeSpaceDataset, MAPE, MSE, MAE

def run_lstm(config_path):
    with open(config_path, 'r') as config_json:
        config = json.load(config_json)

    data_config = config['data_config']
    training_config = config['training_config']
    model_config = config['model_config']
    optimizer_config = config['optimizer_config']
    scheduler_config = config['scheduler_config']

    with open('./data/web-traffic-time-series-forecasting/time_space_matrix.pickle', 'rb') as file:
        time_space_matrix = pickle.load(file)

    train_time_space_matrix = time_space_matrix[:-data_config['val_timesteps_split'], :]
    val_time_space_matrix = time_space_matrix[-data_config['val_timesteps_split']:, :]

    scaler = StandardScaler()
    train_time_space_matrix = scaler.fit_transform(train_time_space_matrix)
    val_time_space_matrix = scaler.fit_transform(val_time_space_matrix)

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(
        model_config['output_targets'], 
        model_config['input_features'], 
        model_config['hidden_size'], 
        model_config['num_layers'],
        model_config['in_timesteps']
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), optimizer_config['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.7)
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
        'avg_train_mse_loss': [],
        'avg_train_mae_loss': [],
        'avg_train_mape_loss': [],
        'avg_val_mse_loss': [],
        'avg_val_mae_loss': [],
        'avg_val_mape_loss': [],
        'epoch_time': [],
    }
    with tqdm(total=training_config['num_epochs']) as pbar:
        early_stopping_patience = training_config['early_stopping_patience']
        early_stopping_delta = training_config['early_stopping_delta']

        for epoch in range(training_config['num_epochs']):
            start = time.time()
            model.train()
            total_train_mse_loss = 0
            total_train_mae_loss = 0
            total_train_mape_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                features, target = batch
                features = features.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float).squeeze()
                
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

                    features = features.to(device, dtype=torch.float)
                    target = target.to(device, dtype=torch.float).squeeze()
                    predict = model(features)

                    predict = scaler.inverse_transform(predict.squeeze().cpu().numpy())
                    target = scaler.inverse_transform(target.squeeze().cpu().numpy())

                    total_val_mse_loss += MSE(target, predict)
                    total_val_mae_loss += MAE(target, predict)
                    total_val_mape_loss += MAPE(target, predict)

            end = time.time()
            training_stats['epoch_time'].append(end-start)

            training_stats['avg_val_mse_loss'].append(total_val_mse_loss/len(val_dataset))
            training_stats['avg_val_mae_loss'].append(total_val_mae_loss/len(val_dataset))
            training_stats['avg_val_mape_loss'].append(total_val_mape_loss/len(val_dataset))
                        
            if (epoch+1)%scheduler_config['decay_epoch']==0:
                scheduler.step()

            if len(training_stats['avg_val_mape_loss']) > 1:
                if training_stats['avg_val_mape_loss'][-1]>training_stats['avg_val_mape_loss'][-2]:
                    early_stopping_patience -= 1
                else:
                    if training_stats['avg_val_mape_loss'][-2]-training_stats['avg_val_mape_loss'][-1]<early_stopping_delta:
                        early_stopping_patience -= 1
                    else:
                        early_stopping_patience = training_config['early_stopping_patience']
                        
            if early_stopping_patience == 0:
                break

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
    wandb.init(project="tugas-akhir", entity="mikhaelbelmiro", name='LSTM')
    config_path = sys.argv[1]
    run_lstm(config_path)