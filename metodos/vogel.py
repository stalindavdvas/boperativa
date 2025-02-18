def metodo_vogel(costos, suministros, demandas):
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
    penalizaciones = []  # Para almacenar las penalizaciones en cada iteración

    while sum(suministros) > 0 and sum(demandas) > 0:
        # Calcular penalizaciones por fila y columna
        penalizacion_fila = []
        penalizacion_columna = []

        for i in range(n):
            fila = [costos[i][j] for j in range(m) if suministros[i] > 0 and demandas[j] > 0]
            if len(fila) >= 2:
                fila.sort()
                penalizacion_fila.append(fila[1] - fila[0])
            else:
                penalizacion_fila.append(0)

        for j in range(m):
            columna = [costos[i][j] for i in range(n) if suministros[i] > 0 and demandas[j] > 0]
            if len(columna) >= 2:
                columna.sort()
                penalizacion_columna.append(columna[1] - columna[0])
            else:
                penalizacion_columna.append(0)

        # Guardar las penalizaciones actuales
        penalizaciones.append({
            'fila': penalizacion_fila,
            'columna': penalizacion_columna
        })

        # Encontrar la mayor penalización (fila o columna)
        max_penalizacion_fila = max(penalizacion_fila) if penalizacion_fila else 0
        max_penalizacion_columna = max(penalizacion_columna) if penalizacion_columna else 0

        if max_penalizacion_fila >= max_penalizacion_columna:
            # Asignar en la fila con mayor penalización
            fila_idx = penalizacion_fila.index(max_penalizacion_fila)
            col_idx = min(
                range(m),
                key=lambda j: costos[fila_idx][j] if suministros[fila_idx] > 0 and demandas[j] > 0 else float('inf')
            )
        else:
            # Asignar en la columna con mayor penalización
            col_idx = penalizacion_columna.index(max_penalizacion_columna)
            fila_idx = min(
                range(n),
                key=lambda i: costos[i][col_idx] if suministros[i] > 0 and demandas[col_idx] > 0 else float('inf')
            )

        # Asignar la cantidad mínima entre el suministro y la demanda
        cantidad = min(suministros[fila_idx], demandas[col_idx])
        asignaciones[fila_idx][col_idx] = cantidad

        # Actualizar suministro y demanda
        suministros[fila_idx] -= cantidad
        demandas[col_idx] -= cantidad

    return asignaciones, penalizaciones