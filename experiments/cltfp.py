import sys
sys.path.append('.')

import os
import time
import json
import wandb
import torch
import pickle
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

from tqdm import tqdm
from models import CLTFPModel
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from stgcn.utils import TimeSpaceDataset, create_metrics_dict, update_metrics_batch, update_metrics_epoch, scheduler_step

def run_clftp(config_path):
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
    model = CLTFPModel(
        model_config['in_timesteps'],
        model_config['out_timesteps'],
        model_config['out_cnn_channels'],
        model_config['first_cnn_kernel'],
        model_config['second_cnn_kernel'],
        model_config['third_cnn_kernel'],
        time_space_matrix.shape[1],
        model_config['lstm_hidden_channels']
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

    epoch_stats, batch_stats = create_metrics_dict(model_config['out_timesteps'])

    torch.cuda.empty_cache()
    with tqdm(total=training_config['num_epochs']) as pbar:
        early_stopping_patience = training_config['early_stopping_patience']
        current_patience = early_stopping_patience
        early_stopping_delta = training_config['early_stopping_delta']
        monitor_weighted_metric = []

        for epoch in range(training_config['num_epochs']):
            start = time.time()
            model.train()

            with tqdm(total=len(train_dataloader), leave=False) as pbar2:
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    features, target = batch

                    features = features.to(device, dtype=torch.float)
                    target = target.to(device, dtype=torch.float)
                    
                    predict = model(features)
                    loss = loss_criterion(predict, target)
 
                    with torch.no_grad():
                        batch_stats = update_metrics_batch(batch_stats, predict, target, 'train', scaler)

                    loss.backward()
                    optimizer.step()
                    pbar2.update(1)
            epoch_stats = update_metrics_epoch(epoch_stats, batch_stats, 'train')

            model.eval()
            with tqdm(total=len(val_dataloader), leave=False) as pbar2:
                for batch in val_dataloader:
                    with torch.no_grad():
                        features, target = batch

                        features = features.to(device, dtype=torch.float)
                        target = target.to(device, dtype=torch.float)

                        predict = model(features)
                        batch_stats = update_metrics_batch(batch_stats, predict, target, 'val', scaler)
                        pbar2.update(1)

            end = time.time()
            epoch_stats = update_metrics_epoch(epoch_stats, batch_stats, 'val')
            epoch_stats['epoch_time'].append(end-start)
                        
            monitor_stat = [(1, key) for key in epoch_stats.keys() if 'val' in key and 'mape' in key]
            current_patience, current_loss = scheduler_step(
                scheduler, 
                early_stopping_patience, 
                current_patience, 
                early_stopping_delta, 
                epoch+1, 
                scheduler_config['decay_epoch'], 
                epoch_stats, 
                monitor_stat
                )
            monitor_weighted_metric.append(current_loss)
            if current_patience == 0:
                break

            pbar_train_loss = epoch_stats['avg_train_loss'][-1]
            pbar_val_loss = epoch_stats['avg_val_loss'][-1]
            pbar.set_description(
                f'epoch: {epoch+1}, '\
                f'avg_train_loss: {round(pbar_train_loss, 3)}, ' \
                f'avg_val_loss: {round(pbar_val_loss, 3)}, '\
                f'patience: {current_patience}'
                )
            pbar.update(1)

            pd.DataFrame(epoch_stats).to_csv(f'{out_dir}/training_results.csv', index=False)

            rank = sorted(monitor_weighted_metric).index(monitor_weighted_metric[-1]) + 1
            if rank < training_config['save_n_best']:
                torch.save(model, f'{out_dir}/model-rank{rank}.pth')

            wandb.log(
                {key: val[-1] for key, val in epoch_stats.items()}
            )
            wandb.log(
                {'monitor': monitor_weighted_metric[-1]}
            )

if __name__ == '__main__':
    load_dotenv()
    WANDB_PROJECT_KEY = os.getenv('WANDB_PROJECT_KEY', '')
    wandb.login(key=WANDB_PROJECT_KEY)
    wandb.init(project="tugas-akhir", entity="mikhaelbelmiro", name='CLFTP')
    config_path = sys.argv[1]
    run_clftp(config_path)