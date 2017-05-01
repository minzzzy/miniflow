import numpy as np

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


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        pass
    
    def backward(self):
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        XX = self.inbound_nodes[0].value
        WW = self.inbound_nodes[1].value
        bb = self.inbound_nodes[2].value
        self.value = np.dot(XX,WW) + bb        

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}  # Initialize to 0
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)  # dL / dX = W_T
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)  # dL / dW = X_T
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)  # dL / db = 1


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])
    
    def _sigmoid(self, x):
        return 1./(1.+np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)  # S = 1 / ( 1 + e**(-x))

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}  # Initialize to 0 
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += grad_cost*self.value*(1-self.value)  # (dC / dS) * (dS / dL)


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1,1)
        a = self.inbound_nodes[1].value.reshape(-1,1)
        self.m = len(y)
        self.error = y-a
        self.value = np.sum(np.square(self.error))/self.m  # C = SIGMA((y-a)**2) / m

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2/self.m)*self.error  # dC / dy  
        self.gradients[self.inbound_nodes[1]] = (-2/self.m)*self.error # dC / da


def topological_sort(feed_dict):
    input_nodes = [ n for n in feed_dict.keys() ]

    G = {}
    nodes = [ n for n in input_nodes ]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = { 'in': set(), 'out': set() }
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = { 'in': set(), 'out': set() }
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_and_backward(graph, train=True):
    for n in graph:
        n.forward()

    if (train):
        for n in graph[::-1]:  # Extended slice - reverse order
            n.backward()

def sgd_update(trainables, learning_rate=0.2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]
