import numpy as np

def one_hot(y):
    x = np.zeros((y.size, y.max()+1))
    x[np.arange(y.size),y] = 1
    return x

def load_mnist(path='./data'):
    train = np.genfromtxt(path+'/mnist_train.csv', skip_header=True, delimiter=',', dtype='int')
    test = np.genfromtxt(path+'/mnist_test.csv', skip_header=True, delimiter=',', dtype='int')
    x_train, y_train = train[:,1:], one_hot(train[:,0])
    x_test, y_test = test[:,1:], one_hot(test[:,0])
    x_train = x_train/255
    x_test = x_test/255
    return x_train, y_train, x_test, y_test

def train_one_epoch(network, feature, target, loss_func, lr, batch_size=1):
    loss = 0
    for i in range(0, len(feature), batch_size):
        x, y_true = feature[i:i+batch_size], target[i:i+batch_size]
        out = x
        for layer in network:
            out = layer.forward(out)
        loss += loss_func(y_true, out)
        delta = loss_func(y_true, out, derv=True)
        for layer in reversed(network):
            delta = layer.backward(delta, lr)
    loss /= len(feature)
    return network, loss

def evaluate(network, feature, target, batch_size=1):
    correct = []
    for i in range(0, len(feature), batch_size):
        x, y_true = feature[i:i+batch_size], target[i:i+batch_size]
        out = x
        for layer in network:
            out = layer.forward(out)
        correct.extend(out.argmax(axis=1)==y_true.argmax(axis=1))
    return np.mean(correct)
