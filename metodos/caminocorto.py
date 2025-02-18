def dijkstra(graph, start, end):
    import heapq

    # Inicializar distancias y cola de prioridad
    distances = {node: float('inf') for node in graph}
    previous_nodes = {node: None for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Si llegamos al nodo final, reconstruir el camino
        if current_node == end:
            path = []
            while current_node:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            return path[::-1], distances[end]

        # Explorar los vecinos del nodo actual
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Si no se encuentra un camino
    return [], float('inf')