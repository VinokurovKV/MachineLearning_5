import numpy as np

def get_part_of_array(X):
    return X[0::4, 120:500:5]

def sum_non_neg_diag(X):
    result = X.diagonal()[X.diagonal() >= 0]
    if (np.size(result) != 0):
        return result.sum()
    else:
        return -1

def replace_values(X):
    X_np = np.array(X, dtype=float)
    
    column_means = np.mean(X_np, axis=0)
    
    mask_high = X_np > 1.5 * column_means
    mask_low = X_np < 0.25 * column_means
    
    X_np[mask_high | mask_low] = -1
    
    return X_np