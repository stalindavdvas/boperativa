def knapsack(items, capacity):
    """
    Resuelve el Problema de la Mochila usando Programación Dinámica.
    Args:
        items: Lista de diccionarios con 'peso' y 'valor'.
        capacity: Capacidad máxima de la mochila.
    Returns:
        max_value: Valor máximo que se puede obtener.
        selected_items: Índices de los elementos seleccionados.
    """
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Llenar la tabla DP
    for i in range(1, n + 1):
        peso = items[i - 1]['peso']
        valor = items[i - 1]['valor']
        for w in range(capacity + 1):
            if peso <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - peso] + valor)
            else:
                dp[i][w] = dp[i - 1][w]

    # Reconstruir la solución
    max_value = dp[n][capacity]
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= items[i - 1]['peso']

    return max_value, selected_items