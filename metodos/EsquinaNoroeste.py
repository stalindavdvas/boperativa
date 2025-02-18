def esquina_noroeste(costos, suministros, demandas):
    # Balanceamos el problema si es necesario
    suma_suministros = sum(suministros)
    suma_demandas = sum(demandas)

    # Si el total de suministros no es igual al total de demandas, balanceamos
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

    # Matriz de soluciones (asignaciones)
    asignaciones = [[0 for _ in range(m)] for _ in range(n)]

    # Inicialización de índices
    i = 0
    j = 0

    # Realizamos las asignaciones
    while i < n and j < m:
        # Se toma el valor mínimo entre el suministro y la demanda
        cantidad = min(suministros[i], demandas[j])

        # Asignamos la cantidad a la matriz de soluciones
        asignaciones[i][j] = cantidad

        # Actualizamos el suministro y la demanda
        suministros[i] -= cantidad
        demandas[j] -= cantidad

        # Si el suministro de la fila actual se agotó, nos movemos a la siguiente fila
        if suministros[i] == 0:
            i += 1

        # Si la demanda de la columna actual se agotó, nos movemos a la siguiente columna
        if demandas[j] == 0:
            j += 1

    return asignaciones


# Ejemplo de uso con un problema desbalanceado
def obtener_resultado(costos, suministros, demandas):
    resultado = esquina_noroeste(costos, suministros, demandas)
    return resultado
