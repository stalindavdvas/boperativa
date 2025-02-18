def kruskal(edges, nodes):
    """
    Calcula el Árbol de Expansión Mínima usando el algoritmo de Kruskal.
    """
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n

        def find(self, u):
            if self.parent[u] != u:
                self.parent[u] = self.find(self.parent[u])  # Path compression
            return self.parent[u]

        def union(self, u, v):
            root_u = self.find(u)
            root_v = self.find(v)

            if root_u == root_v:
                return False

            # Union by rank
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

            return True

    # Mapear nodos a índices numéricos
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Ordenar las aristas por peso
    edges_sorted = sorted(edges, key=lambda edge: edge['weight'])

    uf = UnionFind(len(nodes))
    total_cost = 0
    mst_edges = []

    for edge in edges_sorted:
        u = node_to_index[edge['from']]
        v = node_to_index[edge['to']]
        weight = edge['weight']

        if uf.union(u, v):
            total_cost += weight
            mst_edges.append(edge)

    return total_cost, mst_edges