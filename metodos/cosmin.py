def resolver_transporte_costo_minimo(costos, suministros, demandas):
    """
       Resuelve el problema de transporte usando el método de costo mínimo.
       Args:
           costos: Matriz de costos (n x m).
           suministros: Vector de suministros (n).
           demandas: Vector de demandas (m).
       Returns:
           Un diccionario con la matriz de asignaciones, costo total y prueba de optimalidad.
       """
    # Balancear el problema si es necesario
    suministros, demandas, costos = balancear_problema(suministros, demandas, costos)

    n = len(suministros)  # Número de orígenes
    m = len(demandas)  # Número de destinos

    # Inicializar matriz de asignaciones con ceros
    asignaciones = [[0 for _ in range(m)] for _ in range(n)]

    # Método de costo mínimo
    while sum(suministros) > 0 and sum(demandas) > 0:
        # Encontrar la celda con el menor costo
        min_cost = float('inf')
        min_i, min_j = -1, -1
        for i in range(n):
            for j in range(m):
                if suministros[i] > 0 and demandas[j] > 0 and costos[i][j] < min_cost:
                    min_cost = costos[i][j]
                    min_i, min_j = i, j

        # Asignar la cantidad mínima entre suministro y demanda
        cantidad = min(suministros[min_i], demandas[min_j])
        asignaciones[min_i][min_j] = cantidad

        # Actualizar suministro y demanda
        suministros[min_i] -= cantidad
        demandas[min_j] -= cantidad

    # Manejo de degeneración si es necesario
    agregar_penalizacion(asignaciones)

    # Calcular costo total
    costo_total = sum(asignaciones[i][j] * costos[i][j] for i in range(n) for j in range(m))

    # Prueba de optimalidad
    U, V, es_optima = prueba_optimalidad(asignaciones, costos)

    return {
        "asignaciones": asignaciones,
        "costo_total": costo_total,
        "es_optima": es_optima,
        "U": U,
        "V": V,
    }


def balancear_problema(suministros, demandas, costos):
    """
    Balancea el problema de transporte si la oferta y la demanda no coinciden.
    """
    suma_suministros = sum(suministros)
    suma_demandas = sum(demandas)

    if suma_suministros != suma_demandas:
        if suma_suministros < suma_demandas:
            # Agregar una planta ficticia (suministro adicional)
            suministros.append(suma_demandas - suma_suministros)
            costos.append([0] * len(demandas))
        else:
            # Agregar un destino ficticio (demanda adicional)
            demandas.append(suma_suministros - suma_demandas)
            for row in costos:
                row.append(0)

    return suministros, demandas, costos


def agregar_penalizacion(asignaciones):
    """
    Agrega valores artificiales (ε) en celdas vacías para evitar degeneración.
    Se necesita que haya exactamente (n + m - 1) asignaciones básicas.
    """
    n = len(asignaciones)
    m = len(asignaciones[0])
    num_asignaciones = sum(1 for i in range(n) for j in range(m) if asignaciones[i][j] > 0)
    necesario = n + m - 1

    if num_asignaciones < necesario:
        for i in range(n):
            for j in range(m):
                if asignaciones[i][j] == 0:
                    asignaciones[i][j] = 1e-6  # Agregar un valor artificial pequeño
                    num_asignaciones += 1
                    if num_asignaciones == necesario:
                        return


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

    # Inicializar U y V con None
    U = [None] * n
    V = [None] * m
    U[0] = 0  # Se asigna arbitrariamente el valor 0 a U[0]

    # Calcular U y V iterativamente
    max_iteraciones = n * m
    iteraciones = 0
    while None in U or None in V:
        cambios = False
        for i in range(n):
            for j in range(m):
                if asignaciones[i][j] > 0:  # Celda básica
                    if U[i] is not None and V[j] is None:
                        V[j] = costos[i][j] - U[i]
                        cambios = True
                    elif V[j] is not None and U[i] is None:
                        U[i] = costos[i][j] - V[j]
                        cambios = True
        iteraciones += 1
        if iteraciones > max_iteraciones:
            raise ValueError("No se pudieron calcular todos los valores de U y V debido a una solución degenerada")

        if not cambios:
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