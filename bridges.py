def fill_dict(d, seq, val):
    for key in seq:
        d[key] = val

def read_input():
    return [int(x) for x in input().split()]

def construct_adj_list(edges_num):
    adj_list = {}
    nodes = set()
    for _ in range(edges_num):
        v1, v2 = read_input()
        for v in (v1, v2):
            if v not in nodes:
                nodes.add(v)
            if v not in adj_list:
                adj_list[v] = []
        adj_list[v1].append(v2)
        adj_list[v2].append(v1)
    return adj_list, nodes

def dfs(node, parent, bridges):
    global node_id
    visited.add(node)
    node_id += 1
    low_link_values[node] = node_ids[node] = node_id

    for neighbor in adj_list[node]:
        if neighbor == parent: continue
        if neighbor not in visited:
            dfs(neighbor, node, bridges)
            low_link_values[node] = min(low_link_values[node], low_link_values[neighbor])
            if node_ids[node] < low_link_values[neighbor]:
                bridges.append((node, neighbor))
        else:
            low_link_values[node] = min(low_link_values[node], node_ids[neighbor])

def find_bridges():
    bridges = []
    for node in nodes:
        if node not in visited:
            dfs(node, -1, bridges)
    return bridges

if __name__ == '__main__':
    n, m = read_input()
    adj_list, nodes = construct_adj_list(m)

    node_id = 0
    node_ids = {node: 0 for node in nodes}
    low_link_values = {node: 0 for node in nodes}
    visited = set()

    print(find_bridges())

