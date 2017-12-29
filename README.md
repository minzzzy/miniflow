# Miniflow   
Miniflow is a library to build a small neural networks using NumPy for learning the fundamental abstract of TensorFlow. It is based from [Deep Learning nanodegree foundation](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) in Udacity and updated to my own version.

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


### Graph 
The `topological_sort` function creates graph of neural networks by implementing [topological sorting](https://en.wikipedia.org/wiki/Topological_sorting) using Kahn's Algorithm.    

![Topological_sort](http://www.stoimen.com/blog/wp-content/uploads/2012/10/2.-Topological-Sort.png)  

- Define the graph of nodes and edges.
- Propagation values through the graph.

## How to use
Let's create simple neural network(`simple_nn.py`).   
![simple_nn](./img/simple_nn.png)   

This model is not trained. Check the calculation of gradients with one back propagation.  

- Set node into variable  
```python
X, W, b = Input(), Input(), Input()
y = Input()
l = Linear(X, W, b)
s = Sigmoid(l)
cost = MSE(y, s)

X_ = np.array([[3, 2], [-1, -2]])
W_ = np.array([[2], [3]])
b_ = np.array([-3])
y_ = np.array([[11], [2]])

feed_dict = {
    X: X_,
    W: W_,
    b: b_,
    y: y_
}
```

- Create graph with feed dict
```python
graph = topological_sort(feed_dict)
```

- forward and backward(just one step)
```python
forward_and_backward(graph)
```

- Calculate the gradients of \\( W \\) and \\( b \\)
```python
print(W.gradients[W])
print(b.gradients[b])
```

### Examples
#### Regression - [Boston House Prices dataset](http://scikit-learn.org/stable/datasets/#boston-house-prices-dataset)
#### Classification - [Mnist](http://yann.lecun.com/exdb/mnist/)

