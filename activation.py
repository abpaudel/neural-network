import numpy as np

def sigmoid(x, derv=False):
    if derv:
        return sigmoid(x)*(1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

def tanh(x, derv=False):
    if derv:
        return 1 - np.tanh(x)**2
    return np.tanh(x)

def relu(x, derv=False):
    if derv:
        return np.array(x >= 0).astype('int')
    return np.maximum(0, x)

def lrelu(x, a=0.01, derv=False):
    if derv:
        return np.where(x > 0, np.ones_like(x), a)
    return np.maximum(0,x) + a*np.minimum(0,x)

def elu(x, a=1, derv=False):
    if derv:
        return np.where(x > 0, np.ones_like(x), a * np.exp(x))
    return np.maximum(0,x) + np.minimum(0,a*(np.exp(x)-1))

def selu(x, s=1.0507, a=1.6733, derv=False):
    if derv:
        return np.where(x >= 0, np.ones_like(x)*s, np.exp(x)*a*s)
    return s*elu(x, a=a)
