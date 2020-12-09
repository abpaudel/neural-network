import numpy as np

def mse(y_true, y_pred, derv=False):
    if derv:
        return 2 * (y_pred - y_true) / y_pred.size
    return np.mean(np.power(y_true - y_pred, 2))

def sse(y_true, y_pred, derv=False):
    if derv:
        return y_pred - y_true
    return 0.5 * np.sum(np.power(y_true - y_pred, 2))

def cross_entropy(y_true, y_pred, derv=False):
    # assumes y_pred is output from softmax layer
    if derv:
      return y_pred - y_true
    return np.mean(-y_true * np.nan_to_num(np.log(y_pred)))
    