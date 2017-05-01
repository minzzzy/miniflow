import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *

## Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

## Explore the data
print("The number of (Datasets, Features) = {}".format(X_.shape))
print("The first data is {}".format(X_[0]))
print("The first target data is {}".format(y_[0]), "\n")

## Normalize data
X_ = (X_ -np.mean(X_, axis=0)) / np.std(X_,axis=0)

n_features = X_.shape[1]
n_hidden1 = 26
n_hidden2 = 52
W1_ = np.random.randn(n_features,n_hidden1)
b1_ = np.zeros(n_hidden1)
W2_ = np.random.randn(n_hidden1,n_hidden2)
b2_ = np.zeros(n_hidden2)
W3_ = np.random.randn(n_hidden2,1)
b3_ = np.zeros(1)

## Neural network
X, y = Input(), Input()
W1, b1 = Input(), Input()
W2, b2 = Input(), Input()
W3, b3 = Input(), Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
s2 = Sigmoid(l2)
l3 = Linear(s2, W3, b3)
cost = MSE(y, l3)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_,
    W3: W3_,
    b3: b3_
}

## Hyperparameter
epochs = 2000
learning_rate = 0.01
batch_size = 10

## Total number of examples
m = X_.shape[0]
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainalbes = [W1, b1, W2, b2, W3, b3]

## Train
for i in range(epochs):
    loss =0
    accuracy = 0
    for j in range(steps_per_epoch):
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and Y Inputs
        X.value = X_batch
        y.value = y_batch

        forward_and_backward(graph)

        sgd_update(trainalbes, learning_rate)

        loss += graph[-1].value
        # Accuracy 
        # I am not sure it is right. 
        predict = graph[-2].value
        predict = predict.flatten()
        error = (y.value - predict) < 0.5
        Accuracy = np.sum(error.astype(np.int))/len(y.value) * 100
        accuracy += Accuracy
    accuracy = accuracy/steps_per_epoch
    if (i+1) %10 == 0:
        print ("Epoch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1, loss/steps_per_epoch, accuracy))
