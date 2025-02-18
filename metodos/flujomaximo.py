from collections import defaultdict


def edmonds_karp(graph, source, sink):
    """
    Calcula el flujo máximo usando el algoritmo de Edmonds-Karp.
    """
    def bfs(graph_residual, source, sink, parent):
        visited = set()
        queue = [source]
        visited.add(source)
        parent[source] = None

        while queue:
            current = queue.pop(0)
            for neighbor, capacity in graph_residual[current].items():
                if neighbor not in visited and capacity > 0:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = current
                    if neighbor == sink:
                        return True
        return False

    # Crear el grafo residual
    graph_residual = defaultdict(lambda: defaultdict(int))
    for u in graph:
        for v, capacity in graph[u].items():
            graph_residual[u][v] = capacity

    parent = {}
    max_flow = 0
    used_edges = []

    while bfs(graph_residual, source, sink, parent):
        # Encontrar el flujo mínimo en el camino aumentante
        path_flow = float('inf')
        s = sink
        while s != source:
            path_flow = min(path_flow, graph_residual[parent[s]][s])
            s = parent[s]

        # Actualizar el flujo máximo
        max_flow += path_flow

        # Actualizar el grafo residual
        v = sink
        while v != source:
            u = parent[v]
            graph_residual[u][v] -= path_flow
            graph_residual[v][u] += path_flow

            # Registrar las aristas utilizadas
            used_edges.append({
                'from': u,
                'to': v,
                'capacity': graph[u][v],
                'flow': path_flow
            })

            v = parent[v]

    return max_flow, used_edges