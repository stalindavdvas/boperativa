def calcular_costominimo(costos, suministros, demandas):
    # Balanceamos el problema si es necesario
    suma_suministros = sum(suministros)
    suma_demandas = sum(demandas)

    if suma_suministros != suma_demandas:
        if suma_suministros < suma_demandas:
            suministros.append(suma_demandas - suma_suministros)  # Añadimos fila ficticia
            costos.append([0] * len(demandas))  # Añadimos fila ficticia con costos 0
        else:
            demandas.append(suma_suministros - suma_demandas)  # Añadimos columna ficticia
            for row in costos:
                row.append(0)  # Añadimos columna ficticia con costos 0

    # Inicializamos las matrices de suministro y demanda
    n = len(suministros)
    m = len(demandas)
    asignaciones = [[0 for _ in range(m)] for _ in range(n)]

    # Realizamos las asignaciones basadas en el costo mínimo
    while sum(suministros) > 0 and sum(demandas) > 0:
        # Encontrar la celda con el costo mínimo
        min_cost = float('inf')
        min_i, min_j = -1, -1
        for i in range(n):
            for j in range(m):
                if suministros[i] > 0 and demandas[j] > 0 and costos[i][j] < min_cost:
                    min_cost = costos[i][j]
                    min_i, min_j = i, j

        # Asignar la cantidad mínima entre el suministro y la demanda
        cantidad = min(suministros[min_i], demandas[min_j])
        asignaciones[min_i][min_j] = cantidad

        # Actualizar suministro y demanda
        suministros[min_i] -= cantidad
        demandas[min_j] -= cantidad

    return asignaciones