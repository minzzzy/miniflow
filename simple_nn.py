import numpy as np
from miniflow import *
from pprint import pprint

DEBUG = 0

X, W, b = Input('X'), Input('W'), Input('b')
y = Input('y')
l = Linear(X, W, b, 'l')
s = Sigmoid(l,'s')
cost = MSE(y, s, 'cost')

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

    print("< Graph >")
    for n in graph:
        print(n.name, " : ", n)

    print("< feed_dict >")
    print(X.name, " : ", X.value)
    print(W.name, " : ", W.value) 
    print(b.name, " : ", b.value)
    print(y.name, " : ", y.value) 

    print(cost.name, " : ", cost.value) 
    print("- GRADIENTS in Input : ")
    pprint.pprint(gradients)
