def knapsack_multiple(items, capacity):
    """
    Resuelve el Problema de la Mochila con múltiples unidades de cada elemento.
    Args:
        items: Lista de diccionarios con 'nombre', 'peso', 'valor' y 'cantidad_maxima'.
        capacity: Capacidad máxima de la mochila.
    Returns:
        max_value: Valor máximo que se puede obtener.
        selected_items: Diccionario con la cantidad de cada elemento seleccionado.
    """
    n = len(items)
    dp = [0] * (capacity + 1)  # Tabla DP para almacenar el valor máximo
    selected_items = [0] * n   # Para rastrear cuántas unidades de cada elemento se seleccionan

    for i in range(n):
        peso = items[i]['peso']
        valor = items[i]['valor']
        cantidad_maxima = items[i]['cantidad_maxima']

        # Iterar en orden inverso para evitar sobrescribir valores
        for w in range(capacity, 0, -1):
            for cnt in range(1, cantidad_maxima + 1):
                if cnt * peso <= w:
                    if dp[w] < dp[w - cnt * peso] + cnt * valor:
                        dp[w] = dp[w - cnt * peso] + cnt * valor
                        selected_items[i] = cnt

    # Reconstruir la solución
    max_value = dp[capacity]
    result = {}
    for i in range(n):
        if selected_items[i] > 0:
            result[items[i]['nombre']] = {
                "cantidad": selected_items[i],
                "peso_unitario": items[i]['peso'],
                "valor_unitario": items[i]['valor']
            }

    return max_value, result