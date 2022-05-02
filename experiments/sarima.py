import os
import torch
import pickle

import numpy as np
import pandas as pd


from tqdm import tqdm
from pmdarima.arima import auto_arima
from stgcn.utils import MAPE, MSE, MAE
from sklearn.preprocessing import StandardScaler

def run_sarima():
    with open('./data/web-traffic-time-series-forecasting/time_space_matrix.pickle', 'rb') as file:
        time_space_matrix = pickle.load(file)

    train_time_space_matrix = time_space_matrix[:-60, :]
    val_time_space_matrix = time_space_matrix[-60:, :]

    save_dir = './model/sarima'
    results_path = f'{save_dir}/training_results.csv'
    os.makedirs(save_dir, exist_ok=True)


    if os.path.exists(results_path):
        training_stats = pd.read_csv(results_path).to_dict(orient='list')
    else:
        training_stats = {
        'mape': [],
        'mae': [],
        'mse': []
    }

    with tqdm(total=train_time_space_matrix.shape[1]) as pbar:
        for i in range(train_time_space_matrix.shape[1]):
            pbar.set_description(f'doing {i+1}')

            if i < len(training_stats['mape']):
                pbar.update(1)
                continue
            
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_time_space_matrix[:, i].reshape(-1, 1))
            val_data = val_time_space_matrix[:, i].reshape(-1, 1)

            model = auto_arima(train_data)
            pred = scaler.inverse_transform(model.predict(60).reshape(-1, 1))
            
            training_stats['mape'].append(MAPE(val_data, pred))
            training_stats['mae'].append(MAE(val_data, pred))
            training_stats['mse'].append(MSE(val_data, pred))

            pbar.update(1)
            pd.DataFrame(training_stats).to_csv(f'{save_dir}/training_results.csv', index=False)

    avg_mse_loss = np.mean(training_stats['mse'])
    avg_mae_loss = np.mean(training_stats['mae'])
    avg_mape_loss = np.mean(training_stats['mape'])

    print(f'avg_mse_loss: \t {avg_mse_loss}')
    print(f'avg_mae_loss: \t {avg_mae_loss}')
    print(f'avg_mape_loss: \t {avg_mape_loss}')

if __name__ == '__main__':
    run_sarima()