import pulp


def resolver_transporte_pulp(costos, suministros, demandas):
    """
    Resuelve el problema de transporte usando PuLP.
    Args:
        costos: Matriz de costos (n x m).
        suministros: Vector de suministros (n).
        demandas: Vector de demandas (m).
    Returns:
        Un diccionario con las asignaciones, el costo total y la prueba de optimalidad.
    """
    n = len(suministros)  # Número de orígenes
    m = len(demandas)  # Número de destinos

    # Crear el problema de minimización
    prob = pulp.LpProblem("Problema_de_Transporte", pulp.LpMinimize)

    # Variables de decisión
    x = [[pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous') for j in range(m)] for i in range(n)]

    # Función objetivo: Minimizar el costo total
    prob += pulp.lpSum(costos[i][j] * x[i][j] for i in range(n) for j in range(m))

    # Restricciones de suministro
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(m)) <= suministros[i]

    # Restricciones de demanda
    for j in range(m):
        prob += pulp.lpSum(x[i][j] for i in range(n)) >= demandas[j]

    # Resolver el problema
    prob.solve()

    # Extraer las asignaciones y el costo total
    asignaciones = [[pulp.value(x[i][j]) for j in range(m)] for i in range(n)]
    costo_total = pulp.value(prob.objective)

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
