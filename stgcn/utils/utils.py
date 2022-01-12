import torch
import numpy as np
import time


def normalize_matrix(mat):
    mat = mat + np.eye(mat.shape[0])
    diag = np.diagflat(np.sum(mat, axis=1))
    diag_inv = np.reciprocal(np.sqrt(diag))
    diag_inv[diag_inv>1e9] = 0
    out = diag_inv@mat@diag_inv
    return(out)


