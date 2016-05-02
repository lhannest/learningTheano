import matplotlib.pyplot as plt
from theano import tensor as T
import data.mnist as data
import theano.tensor.nnet as nnet
import theano
import numpy as np

def size(array):
    s = 1
    for v in array.shape:
        s = s * v
    return s

def flatten(array):
    return T.reshape(m, (size(m),)).eval()

def makelayer(X, input_size, output_size):
    w = np.random.randn(input_size + 1, output_size)
    W = theano.shared(np.asarray(w, dtype=theano.config.floatX))
    bias = np.asarray(np.random.randn(1), dtype=theano.config.floatX)
    B = theano.shared(bias)
    new_X = T.concatenate([X, B])
    return nnet.sigmoid(T.dot(W.T, new_X)), W, B


def makeConvlayer(X, kernal_shape, activation=nnet.sigmoid):
    assert len(kernal_shape) == 2, 'this is the shape of the image, it must be 2D.'
    W = theano.shared(np.asarray(np.random.randn(1, 1, *kernal_shape), dtype=theano.config.floatX))
    Z = nnet.conv2d(X, W)
    return activation(Z), W

train_data = data.getTrainingSet()
test_data = data.getTestingSet()

Y_hat = T.vector()
X = T.vector()
weights = []
Y, W, B = makelayer(X, 784, 50)
weights.append(W)
weights.append(B)
Y, W, B = makelayer(Y, 50, 10)
weights.append(W)
weights.append(B)

mse = T.mean(T.sqr(Y - Y_hat))
updates = [(w, w - T.grad(mse, w) * 0.01) for w in weights]

train_model = theano.function(inputs=[X, Y_hat], outputs=[mse], updates=updates)
evaluate_model = theano.function(inputs=[X], outputs=[Y])

# print evaluate_model(np.random.randn(1, 1, 28, 28))
# print train_model(np.random.randn(1, 1, 28, 28), np.random.randn(10))

i = 0
for i in range(10):
    for t, x in train_data:
        x = x.reshape(784)
        e = train_model(x, t)
    print i, e

accuracy = 0
for t, x in test_data:
    x = x.reshape(784)
    y = evaluate_model(x)[0]
    if data.demux(y) == data.demux(t):
        accuracy += 1
print 'accuracy:', accuracy * 100.0 / len(test_data)
