import matplotlib.pyplot as plt
from theano import tensor as T
import theano.tensor.nnet as nnet
import theano
import numpy as np
from PIL import Image

# G = T.grad(cost=T.sum(Y), wrt=X)
# fn = theano.function(inputs=[X, K], outputs=[G])
# print fn(x, k)

# output_shape = (1, 1, 3, 3)
# image = T.alloc(0., *output_shape)
# C = nnet.conv2d(image, K)
# G = T.grad(C.sum(), wrt=image, known_grads={C: X})
# fn = theano.function(inputs=[X, K], outputs=[G])
# print fn(x, k)[0]

dtype=theano.config.floatX

def deconv2d(X, kernal_shape):
    k = np.random.randn(*kernal_shape)
    K = theano.shared(np.asarray(k, dtype=dtype))
    img = T.zeros_like(X)
    C = nnet.conv2d(img, K)
    return T.grad(T.sum(C), wrt=img, known_grads={C: X}), K

def conv2d(X, kernal_shape):
    assert len(kernal_shape) == 4
    k = np.random.randn(*kernal_shape)
    K = theano.shared(np.asarray(k, dtype=dtype))
    return nnet.conv2d(X, K), K

# x = np.random.randn(1, 1, 3, 3).astype('float32') * 10
x = np.eye(5).reshape(1, 1, 5, 5).astype(dtype)
X = T.dtensor4()

kernals = []
Y, K = conv2d(X, (1, 1, 2, 2))
kernals.append(K)
Y, K = deconv2d(Y, (1, 1, 2, 2))
kernals.append(K)

mse = T.mean(T.sqr(Y - X))
updates = [(k, T.cast(k - T.grad(mse, k) * 0.01, dtype)) for k in kernals]

train_model = theano.function(inputs=[X], outputs=[mse], updates=updates)
evaluate_model = theano.function(inputs=[X], outputs=[Y])
for i in range(10000):
    e = train_model(x)
    if i % 100 == 0:
        print str(e)
print x
y = evaluate_model(x)[0]
print y
print np.round(y)
