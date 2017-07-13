import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *
import matplotlib.pyplot as plt

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

## Training & Test datasets
t = int(X_.shape[0]*0.9)
X_train = X_[:t][:]
X_test = X_[t:][:]
y_train = y_[:t][:]
y_test = y_[t:][:]

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
    X: X_train,
    y: y_train,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_,
    W3: W3_,
    b3: b3_
}

## Hyperparameter
epochs = 2000
learning_rate = 0.001
batch_size = 10
error_tol = 0.5

## Total number of examples
m = X_train.shape[0]
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainalbes = [W1, b1, W2, b2, W3, b3]

## Train
print("Start Training...")

for i in range(epochs):
    loss =0
    accuracy = 0
    for j in range(steps_per_epoch):
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_train, y_train, n_samples=batch_size)

        # Reset value of X and Y Inputs
        X.value = X_batch
        y.value = y_batch

        forward_and_backward(graph)

        sgd_update(trainalbes, learning_rate)

        loss += graph[-1].value
        # Accuracy 
        predict = graph[-2].value
        predict = predict.flatten()
        error = np.sum(y.value - predict) < error_tol
        Accuracy = np.sum(error.astype(np.int))/len(y.value) * 100
        accuracy += Accuracy
    accuracy = accuracy/steps_per_epoch
    if (i+1) % batch_size == 0:
        print ("Epoch: {}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1, loss/steps_per_epoch, accuracy))

print("Finish Training!\n")

## Test
feed_dict[X] = X_test
feed_dict[y] = y_test

graph = topological_sort(feed_dict)

forward_and_backward(graph, train=False)
predict = graph[-2].value
predict = predict.flatten()
error = np.sum(y.value - predict) < error_tol
Accuracy = np.sum(error.astype(np.int))/len(y.value) * 100
print("Test Accuracy : {:.3f}".format(Accuracy))

fig, ax = plt.subplots()
x = np.arange(1, X_test.shape[0]+1)
ax.plot(x, predict, label='Prediction')
ax.plot(x, y_test, label='Test date')
ax.legend(loc='upper right', shadow=False)
plt.show()

