import pulp


def resolver_transporte_costo_minimo(costos, suministros, demandas):
    """
    Resuelve el problema de transporte usando el método del costo mínimo.
    Args:
        costos: Matriz de costos (n x m).
        suministros: Vector de suministros (n).
        demandas: Vector de demandas (m).
    Returns:
        Un diccionario con las asignaciones, el costo total y la prueba de optimalidad.
    """
    n = len(suministros)  # Número de orígenes
    m = len(demandas)  # Número de destinos

    # Crear una copia mutable de suministros y demandas
    suministros = suministros[:]
    demandas = demandas[:]

    # Inicializar la matriz de asignaciones
    asignaciones = [[0 for _ in range(m)] for _ in range(n)]

    # Método del costo mínimo
    while any(s > 0 for s in suministros) and any(d > 0 for d in demandas):
        # Encontrar la celda con el menor costo
        min_cost = float('inf')
        min_i, min_j = -1, -1
        for i in range(n):
            for j in range(m):
                if suministros[i] > 0 and demandas[j] > 0 and costos[i][j] < min_cost:
                    min_cost = costos[i][j]
                    min_i, min_j = i, j

        if min_i == -1 or min_j == -1:
            break

        # Asignar la cantidad máxima posible
        cantidad = min(suministros[min_i], demandas[min_j])
        asignaciones[min_i][min_j] = cantidad
        suministros[min_i] -= cantidad
        demandas[min_j] -= cantidad

    # Calcular el costo total
    costo_total = sum(asignaciones[i][j] * costos[i][j] for i in range(n) for j in range(m))

    # Realizar la prueba de optimalidad
    U, V, es_optima = prueba_optimalidad(asignaciones, costos)

    return {
        "asignaciones": asignaciones,
        "costo_total": costo_total,
        "es_optima": es_optima,
        "U": U,
        "V": V,
    }


def prueba_optimalidad(asignaciones, costos):
    """
    Realiza la prueba de optimalidad usando el método MODI (UV).
    Args:
        asignaciones: Matriz de asignaciones (n x m).
        costos: Matriz de costos (n x m).
    Returns:
        U: Valores de U (n).
        V: Valores de V (m).
        es_optima: True si la solución es óptima, False en caso contrario.
    """
    n = len(asignaciones)
    m = len(asignaciones[0])

    # Inicializar U y V
    U = [None] * n
    V = [None] * m
    U[0] = 0  # Asignamos un valor arbitrario a U[0]

    # Calcular U y V
    max_iteraciones = n * m
    iteraciones = 0
    while None in U or None in V:
        for i in range(n):
            for j in range(m):
                if asignaciones[i][j] > 0:  # Celda básica
                    if U[i] is not None and V[j] is None:
                        V[j] = costos[i][j] - U[i]
                    elif V[j] is not None and U[i] is None:
                        U[i] = costos[i][j] - V[j]
        iteraciones += 1
        if iteraciones > max_iteraciones:
            raise ValueError("No se pudieron calcular todos los valores de U y V")

    # Verificar optimalidad
    es_optima = True
    for i in range(n):
        for j in range(m):
            if asignaciones[i][j] == 0:  # Celda no básica
                costo_reducido = costos[i][j] - (U[i] + V[j])
                if costo_reducido < -1e-6:  # Tolerancia para errores numéricos
                    es_optima = False
                    break
        if not es_optima:
            break

    return U, V, es_optima