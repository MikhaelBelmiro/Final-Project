import math
import torch
import numpy as np

from torch.utils.data import Dataset

def process_adj_mat(mat, eps, sigma_sq):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            old_entry = mat[i, j]
            new_entry = math.exp(-(old_entry**2/sigma_sq))
            if new_entry > eps:
                mat[i, j] = new_entry
            else:
                mat[i, j] = 0.0
    return mat

def normalize_matrix(mat):
    mat = mat + np.eye(mat.shape[0])
    diag = np.diagflat(np.sum(mat, axis=1))
    diag_inv = np.reciprocal(np.sqrt(diag))
    diag_inv[diag_inv>1e9] = 0
    out = diag_inv@mat@diag_inv
    return out

class TimeSpaceDataset(Dataset):
    def __init__(self, time_space_matrix, in_timesteps, out_timesteps):
        super().__init__()
        self.matrix = time_space_matrix
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps

    def __len__(self):
        return self.matrix.shape[0] - (self.in_timesteps+self.out_timesteps-1)

    def __getitem__(self, index):
        features =self.matrix[index:index+self.in_timesteps, :]
        target = self.matrix[index+self.in_timesteps:index+self.in_timesteps+self.out_timesteps, :]

        return features, target

def MAPE(v, v_):
    return np.mean(np.abs((v_-v)/v))


def MSE(v, v_):
    return np.mean((v_-v)**2)


def MAE(v, v_):
    return np.mean(np.abs(v_-v))

def create_metrics_dict(out_timesteps):
    epoch_metric = {}
    for i in range(1, out_timesteps+1):
        epoch_metric[f'avg_train_{i}_step_mae_loss'] = []
        epoch_metric[f'avg_train_{i}_step_mape_loss'] = []
        epoch_metric[f'avg_val_{i}_step_mae_loss'] = []
        epoch_metric[f'avg_val_{i}_step_mape_loss'] = []
    general_epoch_stats = {
        'avg_train_loss': [],
        'avg_val_loss': [],
        'epoch_time': []
    }
    epoch_stats = {**epoch_metric, **general_epoch_stats}

    batch_metric = {}
    for i in range(1, out_timesteps+1):
        batch_metric[f'total_train_{i}_step_mae_loss'] = []
        batch_metric[f'total_train_{i}_step_mape_loss'] = []
        batch_metric[f'total_val_{i}_step_mae_loss'] = []
        batch_metric[f'total_val_{i}_step_mape_loss'] = []
    general_batch_stats = {
        'total_train_loss': [],
        'total_val_loss': [],
    }
    batch_stats = {**batch_metric, **general_batch_stats}
    return epoch_stats, batch_stats

def update_metrics_batch(batch_stats, preds, target, mode, scaler=None):
    for i in range(1, preds.shape[1]+1):
        if scaler:
            temp_preds = scaler.inverse_transform(preds[:, i-1, :].cpu().numpy())
            temp_target = scaler.inverse_transform(target[:, i-1, :].cpu().numpy())

        mse = MSE(temp_target, temp_preds)
        mae = MAE(temp_target, temp_preds)
        mape = MAPE(temp_target, temp_preds)

        if mode == 'train':
            batch_stats[f'total_train_{i}_step_mae_loss'].append(mae)
            batch_stats[f'total_train_{i}_step_mape_loss'].append(mape)
            batch_stats['total_train_loss'].append(mse)
        elif mode == 'val':
            batch_stats[f'total_val_{i}_step_mae_loss'].append(mae)
            batch_stats[f'total_val_{i}_step_mape_loss'].append(mape)
            batch_stats['total_val_loss'].append(mse)
            
    return batch_stats

def update_metrics_epoch(epoch_stats, batch_stats, mode):
    for batch_key in batch_stats.keys():
        if mode in batch_key:
            epoch_key = 'avg_' + batch_key.split('total_')[-1]
            epoch_stats[epoch_key].append(np.mean(batch_stats[batch_key]))
    return epoch_stats

def calculate_weighted_monitor_stat(epoch_stats, monitor_stat):
    current_loss = 0
    prev_loss = 0
    for weight, col in monitor_stat:
        current_loss += weight*epoch_stats[col][-1]
        prev_loss += weight*epoch_stats[col][-2]
        total_weight = weight
    current_loss = current_loss/total_weight
    prev_loss = prev_loss/total_weight
    return current_loss, prev_loss

def scheduler_step(scheduler, init_patice, current_patience, delta, current_epoch, decay_epoch, epoch_stats, monitor_stat):
    if isinstance(monitor_stat, str):
        monitor_stat = (1, monitor_stat)
    
    if current_epoch%decay_epoch==0:
        scheduler.step()

    current_loss = 1.0
    if len(epoch_stats['epoch_time']) > 1:
        current_loss, prev_loss = calculate_weighted_monitor_stat(epoch_stats, monitor_stat)

        if current_loss>prev_loss:
            current_patience -= 1
        else:
            if prev_loss-current_loss<delta:
                current_patience -= 1
            else:
                current_patience = init_patice
    return current_patience, current_loss