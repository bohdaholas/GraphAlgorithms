import heapq

MAX_VAL = float('inf')
INF = MAX_VAL / 2

def read_input():
    return [int(x) for x in input().split()]

class Edge:
    def __init__(self, v1, v2, w):
        self.v1 = v1
        self.v2 = v2
        self.w = w

    def __str__(self):
        return f"v1 = {self.v1} v2 = {self.v2} w = {self.w}"

def construct_adj_matrix(nodes_num):
    adj_matrix = [[INF for _ in range(nodes_num + 1)] for _ in range(nodes_num + 1)]
    for i in range(1, nodes_num + 1):
        adj_matrix[i] = read_input()
    return adj_matrix

def construct_edge_list(edges_num):
    edge_list = []
    nodes = set()
    for _ in range(edges_num):
        u, v, w = read_input()
        # nodes may be implicitly replaced
        nodes.add(u)
        nodes.add(v)
        edge_list.append(Edge(u, v, w))
        edge_list.append(Edge(v, u, w))
    return nodes, edge_list

def construct_adj_list(edges_num):
    adj_list = {}
    nodes = set()
    for _ in range(edges_num):
        # u, v = read_input()
        u, v, w = read_input()
        nodes.add(u)
        nodes.add(v)
        if u not in adj_list:
            adj_list[u] = []
        if v not in adj_list:
            adj_list[v] = []
        # adj_list[u].append(v)
        # adj_list[v].append(u)
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
    return adj_list, nodes

def dfs(adj_list, node):
    nodes_stack = [node]
    visited = {node}
    while len(nodes_stack):
        u = nodes_stack.pop()
        print(u)
        for v in adj_list[u]:
            if v not in visited:
                nodes_stack.append(v)
                visited.add(v)

def cmp_dfs(adj_list, components, all_visited, count, node):
    """ Helper function for finding graph components """
    nodes_stack = [node]
    visited = {node}
    all_visited.add(node)
    while len(nodes_stack):
        u = nodes_stack.pop()
        components[u] = count
        for v in adj_list[u]:
            if v not in visited:
                nodes_stack.append(v)
                visited.add(v)
                all_visited.add(v)

def find_components(adj_list):
    """ Find connected components """
    components = {}
    count = 0
    all_visited = set()
    nodes = adj_list.keys()
    for node in nodes:
        if node not in all_visited:
            count += 1
            cmp_dfs(adj_list, components, all_visited, count, node)
    return components, count

def bfs(adj_list, node):
    nodes_queue = [node]
    visited = {node}
    while nodes_queue:
        u = nodes_queue.pop(0)
        print(u)
        for v in adj_list[u]:
            if v not in visited:
                nodes_queue.append(v)
                visited.add(v)

def bfs_sp(adj_list, start):
    """ Find shortest path in an undirected graph """
    nodes_queue = [start]
    visited = {start}
    nodes = list(adj_list.keys())
    prev = {node: None for node in nodes}
    while nodes_queue:
        u = nodes_queue.pop(0)
        for v in adj_list[u]:
            if v not in visited:
                nodes_queue.append(v)
                visited.add(v)
                prev[v] = u
    return prev

def reconstruct_path(prev, s, e):
    """ Used after bfs_sp to find distance or reconstruct path """
    path = []
    node = e
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    if path[0] == s:
        return path
    return []

def dfs_top_sort(i, node, visited, ordering, adj_list):
    visited.add(node)
    for neighbor, d in adj_list[node]:
        if neighbor not in visited:
            i = dfs_top_sort(i, neighbor, visited, ordering, adj_list)
    ordering[i] = node
    return i - 1

def top_sort(adj_list):
    """ Find topological sort """
    nodes = list(adj_list.keys())
    visited = set()
    ordering = [0] * len(nodes)
    i = len(nodes) - 1
    for node in nodes:
        if node not in visited:
            i = dfs_top_sort(i, node, visited, ordering, adj_list)
    return ordering

def kahn(adj_list, nodes, in_degree):
    """ Find lexicographically smallest topological sort """
    pq = []
    for v, d in in_degree.items():
        if d == 0:
            heapq.heappush(pq, v)
    idx = 0
    n = len(nodes)
    order = [0] * n
    while len(pq):
        u = heapq.heappop(pq)
        order[idx] = u
        idx += 1
        for v in adj_list[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                heapq.heappush(pq, v)
    if idx != n:
        return []
    return order

def dag_sp(adj_list, nodes, start):
    """ Find shortest path in directed acyclic graph """
    n = len(nodes)
    order = top_sort(adj_list)
    dist = {node: INF for node in nodes}
    dist[start] = 0
    for i in range(n):
        node = order[i]
        if dist[node] != INF:
            for neighbor, d in adj_list[node]:
                dist[neighbor] = min(dist[neighbor], dist[node] + d)
    return dist

def dag_lp(adj_list, nodes, start):
    """ Find longest path in directed acyclic graph """
    n = len(nodes)
    order = top_sort(adj_list)
    dist = {node: INF for node in nodes}
    dist[start] = 0
    for i in range(n):
        node = order[i]
        if dist[node] != INF:
            for neighbor, d in adj_list[node]:
                dist[neighbor] = min(dist[neighbor], dist[node] + d)
    lp_dist = {node: -dist for node, dist in dist.items()}
    return lp_dist

def dijkstra(adj_list, start):
    """ Find shortest path and distance from starting node to all other nodes  
    in weighed undirected graph (weights > 0) """
    visited = set()
    nodes = list(adj_list.keys())
    dist = {node: INF for node in nodes}
    prev = {node: None for node in nodes}
    dist[start] = 0
    pq = []
    heapq.heappush(pq, (dist[start], start))
    while len(pq):
        min_dist, u = heapq.heappop(pq)
        visited.add(u)
        if dist[u] < min_dist: continue
        for v, uv_dist in adj_list[u]:
            if v in visited: continue
            new_dist = dist[u] + uv_dist
            if new_dist < dist[v]:
                prev[v] = u
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return dist, prev

def find_sp(adj_list, s, e):
    """ Using dijkstra algorithm find shortest path and distance from starting to ending node """
    dist, prev = dijkstra(adj_list, s)
    path = []
    if dist[e] == INF: return path
    node = e
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path, dist[e]

def bellman_ford(nodes, edge_list, start):
    """ Find shortest path from starting node to all other nodes  
    in weighed undirected graph (no restictions on weight) """
    dist = {node: INF for node in nodes}
    dist[start] = 0
    n = len(nodes)
    for _ in range(n - 1):
        for edge in edge_list:
            new_dist = dist[edge.v1] + edge.w
            if new_dist < dist[edge.v2]:
                dist[edge.v2] = new_dist
    for _ in range(n - 1):
        for edge in edge_list:
            new_dist = dist[edge.v1] + edge.w
            if new_dist < dist[edge.v2]:
                dist[edge.v2] = -INF
    return dist

def add_neighbors(adj_list, node, visited, pq):
    """ Helper function used in Prim's algorithm """
    visited.add(node)
    for neighbor, cost in adj_list[node]:
        if neighbor not in visited:
            heapq.heappush(pq, (cost, node, neighbor))

def mst_prim(adj_list):
    """ Find minimum spanning tree and its cost """

    visited = set()

    nodes = list(adj_list.keys())
    n = len(nodes)
    start_node = nodes[0]

    mst_edges = []
    edges_counter = mst_cost = 0
    pq = []
    add_neighbors(adj_list, start_node, visited, pq)
    while len(pq) and edges_counter != n - 1:
        edge = heapq.heappop(pq)
        cost, v1, v2 = edge

        if v2 not in visited:
            mst_edges.append(edge)
            edges_counter += 1

            mst_cost += cost
            add_neighbors(adj_list, v2, visited, pq)
    if edges_counter != n - 1:
        return None
    return mst_cost, mst_edges

def propagate_neg_cycles(dp, nxt, n):
    """ Helper function in floyd_warshall algorithm """
    for k in range(n):
        for i in range(n):
            for j in range(n):
                other = dp[i][k] + dp[k][j]
                if other < dp[i][j]:
                    dp[i][j] = -INF
                    nxt[i][j] = -1

def floyd_warshall(adj_matrix, n):
    dp = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    nxt = [[None for _ in range(n + 1)] for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            dp[i][j] = adj_matrix[i][j]
            if adj_matrix[i][j] != INF:
                nxt[i][j] = j
    for k in range(n):
        for i in range(n):
            for j in range(n):
                other = dp[i][k] + dp[k][j]
                if other < dp[i][j]:
                    dp[i][j] = other
                    nxt[i][j] = nxt[i][k]
    propagate_neg_cycles(dp, nxt, n)
    return dp, nxt

def reconstruct_path_fw(adj_matrix, n, s, e):
    dp, nxt = floyd_warshall(adj_matrix, n)
    path = []
    if dp[s][e] == INF:
        return path
    node = s
    while node != e:
        if node == -1: return None
        path.append(node)
        node = nxt[node][e]
    if nxt[node][e] == -1: return None
    path.append(e)
    return path

if __name__ == '__main__':
    n, m = read_input()
    adj_list, nodes = construct_adj_list(m)
    print(dag_lp(adj_list, nodes, 1))
    # print(find_components(adj_list))
    # print(mst_prim(adj_list))

    # nodes, edge_list = construct_edge_list(m)
    # print(bellman_ford(nodes, edge_list, 0))

    # n = int(input())
    # adj_matrix = construct_adj_matrix(n)
    # print(reconstruct_path_fw(adj_matrix, n, 1, n))
