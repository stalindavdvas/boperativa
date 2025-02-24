import heapq
def success_shortest_path(graph, source, sink, demands):
    """
    Implementación del algoritmo Successive Shortest Path para calcular el flujo de costo mínimo.
    Args:
        graph: Grafo representado como un diccionario de adyacencia.
        source: Nodo fuente.
        sink: Nodo sumidero.
        demands: Diccionario de demandas para cada nodo.
    Returns:
        total_cost: Costo total del flujo.
        flow_edges: Lista de aristas con su flujo asignado.
    """

    def dijkstra_with_potentials(graph, source, potentials):
        """Dijkstra modificado para encontrar el camino más corto con potenciales."""
        distances = {node: float('inf') for node in graph}
        previous = {node: None for node in graph}
        distances[source] = 0
        priority_queue = [(0, source)]

        while priority_queue:
            current_dist, u = heapq.heappop(priority_queue)
            if current_dist > distances[u]:
                continue
            for v, capacity, cost, flow in graph[u]:
                if flow < capacity and distances[v] > distances[u] + cost + potentials[u] - potentials[v]:
                    distances[v] = distances[u] + cost + potentials[u] - potentials[v]
                    previous[v] = (u, capacity, cost, flow)
                    heapq.heappush(priority_queue, (distances[v], v))

        return distances, previous

    # Inicializar variables
    nodes = list(graph.keys())
    potentials = {node: 0 for node in nodes}
    flow_edges = []
    total_cost = 0

    # Construir el grafo residual
    residual_graph = {node: [] for node in graph}
    for u in graph:
        for v, capacity, cost in graph[u]:
            residual_graph[u].append([v, capacity, cost, 0])  # [destino, capacidad, costo, flujo]
            residual_graph[v].append([u, 0, -cost, 0])  # Arista inversa inicialmente con flujo 0

    # Iterar hasta satisfacer todas las demandas
    while any(demands[node] != 0 for node in demands):
        # Encontrar nodos con exceso y déficit de flujo
        excess_nodes = [node for node in demands if demands[node] > 0]
        deficit_nodes = [node for node in demands if demands[node] < 0]

        if not excess_nodes or not deficit_nodes:
            break

        source = excess_nodes[0]
        sink = deficit_nodes[0]

        # Calcular el camino más corto con potenciales
        distances, previous = dijkstra_with_potentials(residual_graph, source, potentials)

        if distances[sink] == float('inf'):
            raise ValueError("No hay un camino factible para satisfacer las demandas.")

        # Actualizar potenciales
        for node in potentials:
            potentials[node] += distances[node]

        # Encontrar la cantidad mínima de flujo a enviar
        path = []
        current = sink
        min_flow = float('inf')
        while current != source:
            prev_node, capacity, cost, flow = previous[current]
            path.append((prev_node, current, capacity, cost))
            min_flow = min(min_flow, capacity - flow)
            current = prev_node

        # Actualizar el flujo en el camino
        for u, v, capacity, cost in path:
            for edge in residual_graph[u]:
                if edge[0] == v:
                    edge[3] += min_flow  # Incrementar flujo en la arista directa
                    break
            for edge in residual_graph[v]:
                if edge[0] == u:
                    edge[3] -= min_flow  # Decrementar flujo en la arista inversa
                    break

        # Actualizar demandas
        demands[source] -= min_flow
        demands[sink] += min_flow

        # Guardar el flujo asignado
        for u, v, capacity, cost in path:
            flow_edges.append({"from": u, "to": v, "flow": min_flow, "cost": cost})
            total_cost += min_flow * cost

    return total_cost, flow_edges