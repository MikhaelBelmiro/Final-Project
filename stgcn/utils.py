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
        target = self.matrix[index+self.in_timesteps:index+self.in_timesteps+1, :]

        return features, target
