import pulp

def resolver_transporte_vogel(costos, suministros, demandas):
    """
    Resuelve el problema de transporte usando el método de Vogel.
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

    while sum(suministros) > 0 and sum(demandas) > 0:
        max_penalizacion = -1
        fila_o_columna = None
        indice = None

        # Calcular penalizaciones para filas
        for i in range(n):
            if suministros[i] > 0:
                costos_fila = sorted([costos[i][j] for j in range(m) if demandas[j] > 0])
                penalizacion = costos_fila[1] - costos_fila[0] if len(costos_fila) >= 2 else (costos_fila[0] if costos_fila else 0)
                if penalizacion > max_penalizacion:
                    max_penalizacion = penalizacion
                    fila_o_columna = "fila"
                    indice = i

        # Calcular penalizaciones para columnas
        for j in range(m):
            if demandas[j] > 0:
                costos_columna = sorted([costos[i][j] for i in range(n) if suministros[i] > 0])
                penalizacion = costos_columna[1] - costos_columna[0] if len(costos_columna) >= 2 else (costos_columna[0] if costos_columna else 0)
                if penalizacion > max_penalizacion:
                    max_penalizacion = penalizacion
                    fila_o_columna = "columna"
                    indice = j

        # Asignar en función de la mayor penalización
        if fila_o_columna == "fila":
            i = indice
            min_cost = float('inf')
            j_min = -1
            for j in range(m):
                if demandas[j] > 0 and costos[i][j] < min_cost:
                    min_cost = costos[i][j]
                    j_min = j
            cantidad = min(suministros[i], demandas[j_min])
            asignaciones[i][j_min] += cantidad
            suministros[i] -= cantidad
            demandas[j_min] -= cantidad

        elif fila_o_columna == "columna":
            j = indice
            min_cost = float('inf')
            i_min = -1
            for i in range(n):
                if suministros[i] > 0 and costos[i][j] < min_cost:
                    min_cost = costos[i][j]
                    i_min = i
            cantidad = min(suministros[i_min], demandas[j])
            asignaciones[i_min][j] += cantidad
            suministros[i_min] -= cantidad
            demandas[j] -= cantidad

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
            # Evitar bloqueo forzando valores arbitrarios
            for i in range(n):
                if U[i] is None:
                    U[i] = 0
            for j in range(m):
                if V[j] is None:
                    V[j] = 0
            break

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
