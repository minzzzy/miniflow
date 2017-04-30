# Miniflow  

Miniflow creates a small neural networks using NumPy for learning the fundamental abstract of TensorFlow.  
This is the contents of [Deep Learning nanodegree foundation](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) in Udacity.  
  
## Architecture
It uses a Python class(`miniflow.py`) to represent a generic node.    
Input, Linear, Sigmoid and MSE are subclasses of Node.  
They have forward and backward steps.  

### Graph 
The topological_sort() function creates graph of neural networks by implementing [topological sorting](http://pooh-explorer.tistory.com/51) using Kahn's Algorithm.     
![Topological_sort](http://www.stoimen.com/blog/wp-content/uploads/2012/10/2.-Topological-Sort.png)

## Simple neural network
The architecture of simple neural network(`simple_nn.py`).   
![simple_nn](./img/simple_nn.png =150x200 )   

```python
from miniflow import *

X, W, b = Input(), Input(), Input()
y = Input()
l = Linear(X, W, b)
s = Sigmoid(l)
cost = MSE(y, s)
```
  
## Boston House Prices dataset
I train the network to use the Boston Housing dataset(`boston_nn.py`).  

```shell
Epoch: 10, Loss: 11.437, Accuracy: 59.000
Epoch: 20, Loss: 10.335, Accuracy: 61.000
Epoch: 30, Loss: 7.920, Accuracy: 59.000
Epoch: 40, Loss: 8.254, Accuracy: 58.200
Epoch: 50, Loss: 7.278, Accuracy: 60.800
Epoch: 60, Loss: 8.962, Accuracy: 64.000
Epoch: 70, Loss: 6.686, Accuracy: 60.600
Epoch: 80, Loss: 6.665, Accuracy: 59.800
Epoch: 90, Loss: 6.125, Accuracy: 62.600
Epoch: 100, Loss: 6.556, Accuracy: 58.200
```
