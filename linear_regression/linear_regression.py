# Following: https://www.youtube.com/watch?v=BuIsI-YHzj8
import matplotlib.pyplot as plt
from theano import tensor as T
import theano
import numpy as np

def plot(model, X, M, B, train_x, train_y, color, label):
    output = [model(X, M, B).eval({X: x}) for x in train_x]
    plt.scatter(train_x, train_y, color=color, label=label)
    plt.scatter(train_x, output, color='red')
    # plt.show()

def makeline(data):
    return 5 + (2 * data + np.random.randn(*data.shape) * 0.33)

train_x = np.linspace(0, 1, 100)
train_y = makeline(train_x)

test_x = np.linspace(1, 2, 100)
test_y = makeline(test_x)

# A scalar that will represent each point of train_x
X = T.scalar()
# A scalar that will represent each point of train_y
Y = T.scalar()

# This is our model. Remember, a line is defined as f(x) = m * x + b
# In machine learning language, B is also known as the 'bias'. It allows our model to be transposed on the y-axis!
def model(X, M, B):
    return (X * M) + B

# A weight, initilaized to zero
W0 = theano.shared(np.asarray(0, dtype=theano.config.floatX))
W1 = theano.shared(np.asarray(0, dtype=theano.config.floatX))

# Here we're creating a variable to represent our model (remember, this is a sybolic representation, nothing's actually being calculated yet!)
y = model(X, W0, W1)

# here we define the mean squared error cost function. Here y is the result of our model, and Y is the target value for y
mse = T.mean(T.sqr(y - Y))

# Here we're creating the derivative of the mean squared error function with respect to our weight. This is some kind of sorcery!
gradientW0 = T.grad(cost=mse, wrt=W0)
gradientW1 = T.grad(cost=mse, wrt=W1)

# TODO: Figure this out.
updatesW0 = [[W0, W0 - gradientW0 * 0.01]]
updatesW1 = [[W1, W1 - gradientW1 * 0.01]]

# Building a function that will take a step of stochastic gradient descent each time it's called
trainW0 = theano.function(inputs=[X, Y], outputs=mse, updates=updatesW0)
trainW1 = theano.function(inputs=[X, Y], outputs=mse, updates=updatesW1)

figure = plt.figure()
# No we perform perform stocastic gradient descent on the training set, and do it 100 times!
for i in range(5):
    print i
    # Plot the model each time
    plot(model, X, W0, W1, train_x, train_y, 'blue', 'training')
    plot(model, X, W0, W1, test_x, test_y, 'green', 'testing')
    plt.legend(loc='upper left')
    plt.savefig('figure' + str(i) + '.jpg')
    plt.close()

    for x, y in zip(train_x, train_y):
        trainW0(x, y)
        trainW1(x, y)

# Plot the final result!
plot(model, X, W0, W1, train_x, train_y, 'blue', 'training')
plot(model, X, W0, W1, test_x, test_y, 'green', 'testing')
plt.legend(loc='upper left')
plt.show()
