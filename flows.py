from pprint import pprint

inf = float('inf')

def read_input():
    return [int(x) for x in input().split()]

class Edge:
    def __init__(self, v1, v2, capacity):
        self.v1 = v1
        self.v2 = v2
        self.residual = None
        self.capacity = capacity
        self.flow = 0

    def set_residual(self, residual):
        self.residual = residual

    def remaining_capacity(self):
        return self.capacity - self.flow

    def augment(self, bottleneck):
        self.flow += bottleneck
        self.residual.flow -= bottleneck

    def __str__(self):
        return f"{self.v1} -> {self.v2} {self.flow}/{self.capacity}"

    def __repr__(self):
        return str(self)

def construct_adj_list(edges_num):
    adj_list = {}
    nodes = set()
    for _ in range(edges_num):
        v1, v2, capacity = read_input()
        e1 = Edge(v1, v2, capacity)
        e2 = Edge(v2, v1, 0)
        e1.set_residual(e2)
        e2.set_residual(e1)
        for v in (v1, v2):
            nodes.add(v)
            if v not in adj_list:
                adj_list[v] = []
        adj_list[v1].append(e1)
        adj_list[v2].append(e2)
    return adj_list, nodes

def fill_dict(d, seq, val):
    for key in seq:
        d[key] = val

def dinics():
    level = {}
    nxt = {}
    max_flow = 0
    while bfs(level):
        fill_dict(nxt, nodes, 0)
        while True:
            flow = dfs(s, nxt, inf, level)
            if flow == 0:
                break
            max_flow += flow

    pprint(nxt)
    pprint(level)
    return max_flow

def bfs(level):
    fill_dict(level, nodes, -1)
    q = [s]
    level[s] = 0
    while len(q):
        node = q.pop(0)
        for edge in adj_list[node]:
            if edge.remaining_capacity() > 0 and level[edge.v2] == -1:
                level[edge.v2] = level[node] + 1
                q.append(edge.v2)
    return level[t] != -1

def dfs(node, nxt, flow, level):
    if node == t: return flow
    edges = adj_list[node]
    edges_num = len(edges)
    while nxt[node] < edges_num:
        edge = edges[nxt[node]]
        if edge.remaining_capacity() > 0 and level[edge.v2] == level[node] + 1:
            bottleneck = dfs(edge.v2, nxt, min(flow, edge.remaining_capacity()), level)
            if bottleneck > 0:
                edge.augment(bottleneck)
                return bottleneck
        nxt[node] += 1
    return 0

if __name__ == '__main__':
    n, m = read_input()
    adj_list, nodes = construct_adj_list(m)
    s, t = 1, n

    visited_token = 1
    visited = {node: 0 for node in nodes}

    print(dinics())
