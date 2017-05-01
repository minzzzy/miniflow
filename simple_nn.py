import numpy as np
from miniflow import *
import pprint

DEBUG = 0

X, W, b = Input(), Input(), Input()
y = Input()
l = Linear(X, W, b)
s = Sigmoid(l)
cost = MSE(y, s)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2.],[3.]])
b_ = np.array([-3.])
y_ = np.array([1,2])

feed_dict = {
    X: X_,        
    W: W_,        
    b: b_,        
    y: y_,        
}

graph = topological_sort(feed_dict)

forward_and_backward(graph)

gradients = [t.gradients[t] for t in [X, y, W, b]]

if (DEBUG):
    print("< Nodes >")
    print("- X : ", X)
    print("- W : ", W)
    print("- b : ", b)
    print("- y : ", y)
    print("- l : ", l)
    print("- Sigmoid : ", s)
    print("- cost : ", cost)
    print("")

    print("< Graph >")
    pprint.pprint(graph)
    print("")

    print("< feed_dict >")
    print("- X : ", X.value)
    print("- W : ", W.value)
    print("- b : ", b.value)
    print("- y : ", y.value)
    print("")

    print("- COST : ", cost.value)
    print("- GRADIENTS in Input : ")
    pprint.pprint(gradients)
