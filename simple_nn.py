import numpy as np
from miniflow import *
import pprint

X, W, b = Input(), Input(), Input()
y = Input()
l = Linear(X, W, b)
s = Sigmoid(l)
cost = MSE(y, s)
#print("- X : ", X)
#print("- y : ", y)
#print("- f : ", f)
#print("- Sigmoid : ", s)
#print("- cost : ", cost)

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

#print(X.gradients)
#print(W.gradients)
pprint.pprint(f.gradients)
gradients = [t.gradients[t] for t in [X, y, W, b]]

#print(cost.value)
#print (gradients)
