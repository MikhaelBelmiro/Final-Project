import sys
sys.path.append('.')

import os
import time
import pickle

import pandas as pd

from tqdm import tqdm
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
from stgcn.utils import MAE, MAPE

def run_sarima():
    with open('./data/web-traffic-time-series-forecasting/time_space_matrix.pickle', 'rb') as file:
        time_space_matrix = pickle.load(file)

    train_time_space_matrix = time_space_matrix[:-7, :]
    val_time_space_matrix = time_space_matrix[-7:, :]

    save_dir = './model/sarima'
    results_path = f'{save_dir}/training_results.csv'
    os.makedirs(save_dir, exist_ok=True)


    if os.path.exists(results_path):
        metrics = pd.read_csv(results_path).to_dict(orient='list')
    else:
        metrics = {}
        for i in range(1, 8):
            metrics[f'avg_val_{i}_step_mae_loss'] = []
            metrics[f'avg_val_{i}_step_mape_loss'] = []

    total_training_time = 0
    with tqdm(total=train_time_space_matrix.shape[1]) as pbar:
        for i in range(train_time_space_matrix.shape[1]):
            start = time.time()
            pbar.set_description(f'doing {i+1}')

            if i < len(metrics[f'avg_val_1_step_mae_loss']):
                pbar.update(1)
                continue
            
            scaler = StandardScaler()
            train_data = scaler.fit_transform(train_time_space_matrix[:, i].reshape(-1, 1))
            val_data = val_time_space_matrix[:, i].reshape(-1, 1)

            model = auto_arima(train_data)
            pred = scaler.inverse_transform(model.predict(7).reshape(-1, 1))
            
            for i in range(1, 8):
                metrics[f'avg_val_{i}_step_mae_loss'].append(MAE(val_data[i-1, :], pred[i-1, :]))
                metrics[f'avg_val_{i}_step_mape_loss'].append(MAPE(val_data[i-1, :], pred[i-1, :]))

            pbar.update(1)
            pd.DataFrame(metrics).to_csv(f'{save_dir}/training_results.csv', index=False)
            end = time.time()

            total_training_time = total_training_time + (end-start)

            with open(f'{save_dir}/training_time.txt', 'w') as file:
                file.write(str(total_training_time))

if __name__ == '__main__':
    run_sarima()