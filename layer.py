import numpy as np

class Layer:
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    def __repr__(self):
        pass
    
    def __str__(self):
        return self.__repr__()

class DenseLayer(Layer):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.w = np.random.randn(in_size, out_size) * np.sqrt(1/(in_size + out_size))
        self.b = np.random.randn(1, out_size) * np.sqrt(1/(in_size + out_size))

    def forward(self, inp):
        self.inp = inp
        return np.dot(inp, self.w) + self.b

    def backward(self, delta, lr):
        in_delta = np.dot(delta, self.w.T)
        dw = np.dot(self.inp.T, delta)
        self.w -= lr * dw
        self.b -= lr * delta
        return in_delta
    
    def __repr__(self):
        return f'DenseLayer({self.in_size}, {self.out_size})'

class ActivationLayer(Layer):
    def __init__(self, activation):
        self.activation = activation
    
    def forward(self, inp):
        self.inp = inp
        return self.activation(inp)
    
    def backward(self, delta, lr):
        return delta * self.activation(self.inp, derv=True)
    
    def __repr__(self):
        return f'ActivationLayer({self.activation.__name__})'

class SoftmaxLayer(Layer):
    def __init__(self, in_size):
        self.in_size = in_size
    
    def forward(self, inp):
        inp = inp - np.max(inp)
        exp = np.exp(inp)
        self.out = exp / np.sum(exp)
        return self.out
    
    def backward(self, delta, lr, ce_loss=True):
        if ce_loss: # if cross_entropy loss is used
            return delta
        out = np.tile(self.out.T, self.in_size)
        return self.out * np.dot(delta, np.identity(self.in_size) - out)

    def __repr__(self):
        return f'SoftmaxLayer({self.in_size})'

class FlattenLayer(Layer):
    def __init__(self):
        self.in_shape = None

    def forward(self, inp):
        self.in_shape = inp.shape
        return np.reshape(inp, (1, -1))
    
    def backward(self, delta, lr):
        return np.reshape(delta, self.in_shape)

    def __repr__(self):
        return f'FlattenLayer()'
