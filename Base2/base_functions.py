def get_part_of_array(X):
    result = []
    n = len(X)
    for i in range(0, n, 4):
        row = []
        for j in range(120, 500, 5):
            row.append(X[i][j])
        result.append(row)
    
    return result

def sum_non_neg_diag(X):
    n = len(X)
    m = len(X[0])
    min_dim = min(n, m)
    
    diagonal_sum = 0
    found_positive = False
    
    for i in range(min_dim):
        element = X[i][i]
        if element >= 0:
            diagonal_sum += element
            found_positive = True
    
    return diagonal_sum if found_positive else -1

import copy

def replace_values(X):
    X_copy = copy.deepcopy(X)
    n = len(X)
    m = len(X[0])
    
    for j in range(m):
        column_sum = 0
        for i in range(n):
            column_sum += X[i][j]
        column_mean = column_sum / n
        
        for i in range(n):
            if X_copy[i][j] > 1.5 * column_mean or X_copy[i][j] < 0.25 * column_mean:
                X_copy[i][j] = -1
    
    return X_copy