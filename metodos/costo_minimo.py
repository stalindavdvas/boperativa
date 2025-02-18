import numpy as np


def calcular_costo_minimo(costos, suministros, demandas):
    # Convertir los costos a tipo float para manejar el infinito correctamente
    costos = np.array(costos, dtype=float)

    # Verificar si el problema está desbalanceado
    suma_suministros = sum(suministros)
    suma_demandas = sum(demandas)

    if suma_suministros != suma_demandas:
        # Si desbalanceado, agregar una oferta o demanda ficticia
        if suma_suministros > suma_demandas:
            demandas.append(suma_suministros - suma_demandas)
            costos.append([0] * len(costos[0]))
        elif suma_demandas > suma_suministros:
            suministros.append(suma_demandas - suma_suministros)
            for fila in costos:
                fila.append(0)

    # Convertir a matrices numpy para facilitar el manejo
    suministros = np.array(suministros)
    demandas = np.array(demandas)

    # Inicializar la solución de asignación
    asignaciones = np.zeros_like(costos)

    # Método de Costo Mínimo
    total_suministros = len(suministros)
    total_demandas = len(demandas)

    while suministros.sum() > 0 and demandas.sum() > 0:
        # Encontrar el mínimo costo
        i, j = np.unravel_index(costos.argmin(), costos.shape)

        # Asignar la cantidad mínima entre la oferta y la demanda
        cantidad = min(suministros[i], demandas[j])
        asignaciones[i][j] = cantidad

        # Actualizar suministros y demandas
        suministros[i] -= cantidad
        demandas[j] -= cantidad

        # Si se ha satisfecho la oferta o demanda, eliminar la fila o columna respectiva
        if suministros[i] == 0:
            costos[i, :] = np.inf  # Marcar la fila como procesada
        if demandas[j] == 0:
            costos[:, j] = np.inf  # Marcar la columna como procesada

    # Eliminar el nodo ficticio si se había agregado
    if suma_suministros != suma_demandas:
        if suma_suministros > suma_demandas:
            asignaciones = asignaciones[:-1, :]
        elif suma_demandas > suma_suministros:
            asignaciones = asignaciones[:, :-1]

    return asignaciones.tolist()
