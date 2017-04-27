import sys

class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.outbound_nodes = []
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def __init__(self, *input):
        Node.__init__(self, input)
    
    def forward(self):
        self.value = 0
        for n in range(len(self.inbound_nodes)):
            self.value += self.inbound_nodes[n].value

class Mul(Node):
    def __init__(self, *input):
        Node.__init__(self, input)

    def forward(self):
        self.value = 1
        for n in range(len(self.inbound_nodes)):
            self.value *= self.inbound_nodes[n].value

class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])
    def forward(self):
        inp = self.inbound_nodes[0].value
        wei = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        #self.value = np.dot(inp,wei) + b        

def topological_sort(feed_dict):
    input_nodes = [ n for n in feed_dict.keys() ]
    #print (feed_dict)

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
            G[m]['out'].add(n)
            nodes.append(m)

    #print (G)
    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input):
            n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['out'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    #print (G)
    #print (L)
    #sys.exit()
    return L

def forward_pass(output_node, sorted_nodes):
    for n in sorted_nodes:
        n.forward()
    return output_node.value

if __name__ == "__main__":
    x, y = Input(), Input()
    add = Add(x,y)
    feed_dict = {x:10, y:20}
    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(add, sorted_nodes)
    print ("{} + {} = {} (according to mingflow)".format(feed_dict[x], feed_dict[y], output))








