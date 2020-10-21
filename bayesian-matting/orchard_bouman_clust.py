import numpy as np

class Node(object):
    """
    Params:
        S: np.array(), (N, 3)
        w: np.array(), (N, )
    """
    def __init__(self, S, w):
        W = np.sum(w)
        self.w = w
        self.X = S
        self.left = None 
        self.right = None
        self.mu = np.einsum('ij,i->j', self.X, w) / W  #(3, )
        diff = self.X - np.tile(self.mu, [self.X.shape[0], 1])  #(N, 3)
        t = np.einsum('ij,i->ij', diff, np.sqrt(w))  #(N, 3)
        self.conv = (t.T @ t) / W + 1e-5 * np.eye(3)  #(3, 3)
        self.N = self.X.shape[0]
        V, D = np.linalg.eig(self.conv) #V: (3, ), D: (3, 3)
        self.lmbda = np.max(np.abs(V))
        self.e = D[np.argmax(np.abs(V))]  # (3, )

def split(nodes):
    idx_max = max(enumerate(nodes), key=lambda x: x[1].lmbda)[0]
    C_i = nodes[idx_max]
    idx = C_i.X @ C_i.e <= np.dot(C_i.mu, C_i.e)  # (N, )
    C_a = Node(C_i.X[idx], C_i.w[idx])
    C_b = Node(C_i.X[np.logical_not(idx)], C_i.w[np.logical_not(idx)])
    nodes.pop(idx_max)
    nodes.append(C_a)
    nodes.append(C_b)
    return nodes

def clustFunc(S, w, minVar=0.05):
    mu, sigma = [], []
    nodes = []
    nodes.append(Node(S, w))
    while max(nodes, key=lambda x: x.lmbda).lmbda > minVar:
        nodes = split(nodes)

    for i, node in enumerate(nodes):
        mu.append(node.mu)
        sigma.append(node.conv)

    return np.array(mu), np.array(sigma)
