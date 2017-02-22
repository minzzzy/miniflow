#! /usr/bin/python

import numpy as np
from miniflow import *

#x, y, z = Input(), Input(), Input()
#
#add = Add(x,y,z)
#mul = Mul(x,y,z)
#
#feed_dict = {x:10, y:20, z: 4}
#graph = topological_sort(feed_dict)
#output = forward_pass(add, graph)

#print ("{} + {} + {} = {} (according to mingflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))

X, W, b = Input(), Input(), Input()
y = Input()
f = Linear(X, W, b)
a = Sigmoid(f)
cost = MSE(y, a)

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

print (gradients)
