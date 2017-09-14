# Miniflow  

Miniflow is a library to build a small neural networks using NumPy for learning the fundamental abstract of TensorFlow. 
It is from [Deep Learning nanodegree foundation](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) in Udacity and I modify it to my own version, called Mingflow.  

- version : python 3.6

## Architecture
It uses a Python class to represent a generic node(`Node`). The nodes perform both their own calculations and those of input edges.  

```python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes  # Node(s) from which this Node receives values
        self.outbound_nodes = []   # Node(s) to which this Node passes values
        self.gradients = {}
        # For each inbound Node here, add this Node as an outbound Node to _that_ Node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        self.value = None  # A calculated value

    # Placeholder
    def forward(self):
        # Compute output value based on 'inbound_nodes' and store the result in self.value.
        raise NotImplemented  # Subclasses must implement this function to avoid errors.
    

    # Placeholder
    def backward(self):
        raise NotImplemented  # Subclasses must implement this function to avoid errors.
```

Input, Linear, Sigmoid and MSE are subclasses of Node.   
They have forward and backward steps.  

### Graph 
The topological_sort() function creates graph of neural networks by implementing [topological sorting](http://pooh-explorer.tistory.com/51) using Kahn's Algorithm.     
![Topological_sort](http://www.stoimen.com/blog/wp-content/uploads/2012/10/2.-Topological-Sort.png)  

- Define the graph of nodes and edges.
- Propagation values through the graph.

## Simple neural network
The architecture of simple neural network(`simple_nn.py`).   
![simple_nn](./img/simple_nn.png)   

This model is not trained. Check the calculation of gradients with one back propagation.  

```python
from miniflow import *

X, W, b = Input(), Input(), Input()
y = Input()
l = Linear(X, W, b)
s = Sigmoid(l)
cost = MSE(y, s)
```
  
## [Boston House Prices dataset](http://scikit-learn.org/stable/datasets/#boston-house-prices-dataset)
I train the network to use the Boston Housing dataset(`boston_nn.py`).  
The model has  2 hidden layers and hyperparameters are
> epochs = 5000  
learning_rate = 0.01  
batch_size = 10  

- Train  
> Epoch: 5000, Loss: 0.015

- Test  
> Loss: 30.684  
  
Overfitting!  
It may be due to the small number of data compared to the number of features.
- dataset : 506
- feature : 13
