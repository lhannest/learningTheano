import matplotlib.pyplot as plt
from theano import tensor as T
import theano.tensor.nnet as nnet
import theano
import numpy as np

def leaky(x):
    return T.switch(x < 0, 0.01 * x, x)

def cross_entropy(x, z):
    return -T.sum(x * T.log(x) + (1 - x) * T.log(1 - z), axis=1)

def layer(X, activation_function, input_size, output_size):
    w = np.random.randn(input_size + 1, output_size)
    W = theano.shared(np.asarray(w, dtype=theano.config.floatX))
    bias = np.asarray(np.random.randn(1), dtype=theano.config.floatX)
    return activation_function(T.dot(W.T, T.concatenate([X, bias]))), W

def gradient_descent(cost, w, step_size):
    return w - (T.grad(cost, wrt=w) * step_size)

train_x = np.linspace(-np.pi, np.pi, 100)
train_y = np.sin(train_x)

step_size = 0.1
inputs = T.vector()
targets = T.vector()
y1, W1 = layer(inputs, leaky, 2, 5)
outputs, W3 = layer(y1, leaky, 5, 1)
mse = T.sum((outputs - targets)**2)
train_model = theano.function(
    inputs=[inputs, targets],
    outputs=[mse],
    updates=[
        (W1, gradient_descent(mse, W1, step_size)),
        (W3, gradient_descent(mse, W3, step_size))
    ]
)

evaluate_model = theano.function(inputs=[inputs], outputs=outputs)

train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 2)
train_t = np.array([[0], [1], [1], [0]])

for i in range(1000):
    for x, t in zip(train_x, train_t):
        error = train_model(x, t)
    if i % 100 == 0:
        print i/100, error

for x, t in zip(train_x, train_t):
    print x, int(round(evaluate_model(x), 1))
