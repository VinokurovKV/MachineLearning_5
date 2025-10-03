import numpy as np

def are_multisets_equal(x, y):
  if len(x) != len(y):
    return False
  return np.array_equal(np.sort(x), np.sort(y))

def max_prod_mod_3(x):
    answer = (x[:-1] * x[1:])[(x[:-1] % 3 == 0) | (x[1:] % 3 == 0)]
    return np.max(answer) if (answer.size > 0) else -1

def convert_image(image, weights):
    weights_array = np.array(weights)
    
    result = np.sum(image * weights_array, axis=2)
    
    return result

def rle_scalar(x, y):
    x_repeat = np.repeat(x[:, 0], x[:, 1])
    y_repeat = np.repeat(y[:, 0], y[:, 1])
    if x_repeat.shape != y_repeat.shape:
        return -1
    else:
        return np.dot(x_repeat, y_repeat)

def cosine_distance(X, Y):
    with np.errstate(divide='ignore', invalid='ignore'):
        ans = np.dot(X, Y.T) / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    return np.nan_to_num(ans, nan=1)