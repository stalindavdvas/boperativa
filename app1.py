from flask import Flask, request, jsonify
import numpy as np
import pulp
import json
from metodos.EsquinaNoroeste import obtener_resultado
from metodos.arbolminimo import kruskal
from metodos.big_m import BigMWithSteps
from metodos.caminocorto import dijkstra
from metodos.costo_minimo import calcular_costo_minimo
from metodos.costominimo import calcular_costominimo
from metodos.dual import DualSimplex
from metodos.flujomaximo import edmonds_karp
from metodos.simplex import Simplex
from collections import defaultdict
from flask_cors import CORS

from metodos.two_phase import TwoPhaseWithSteps
from metodos.vogel import metodo_vogel
#import google.generativeai as genai
from google import genai
app = Flask(__name__)
CORS(app)
# Configura tu API key de Gemini
client = genai.Client(api_key="AIzaSyCpzD4M30B2Yx6p8XwCBcDYzdoYxB-24p4")
############### METODO SIMPLEX ##########################################
@app.route('/simplex', methods=['POST'])
def simplex_solver():
    try:
        data = request.json
        c = np.array(data['c'])
        A = np.array(data['A'])
        b = np.array(data['b'])
        problem_type = data.get('problem_type', 'max')
        problem_description = data.get('problem_description', '')

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia de Simplex
        simplex = SimplexWithSteps(c, A, b, problem_type)
        solution, optimal_value, steps, variable_names = simplex.solve_with_steps()

        # Generar el análisis de sensibilidad con Gemini
        sensitivity_analysis = {
            "c_ranges": simplex.get_c_ranges(),
            "b_ranges": simplex.get_b_ranges()
        }

        # Crear el mensaje para Gemini
        gemini_prompt = f"""
        Contexto del problema: {problem_description}
        Resultados del cálculo:
        - Solución óptima: {solution}
        - Valor óptimo: {optimal_value}
        - Análisis de sensibilidad: {sensitivity_analysis}
        Por favor, proporciona un análisis de sensibilidad basado en los resultados anteriores basado en el contexto del problema.
        """

        # Llamar a Gemini
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt,
        )

        interpretation = gemini_response.text
        print(interpretation)

        # Devolver la solución, el valor óptimo, las tablas intermedias, los nombres de las variables y la interpretación
        return jsonify({
            "solution": solution.tolist(),
            "optimal_value": optimal_value,
            "steps": [step.tolist() for step in steps],  # Convertir arrays NumPy a listas
            "variable_names": variable_names,
            "sensitivity_analysis": sensitivity_analysis,
            "interpretation": interpretation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

class SimplexWithSteps(Simplex):
    def solve_with_steps(self):
        """
        Resuelve el problema de programación lineal usando el método simplex
        y captura las tablas intermedias.
        """
        steps = []  # Lista para almacenar las tablas intermedias

        while True:
            # Guardar el tableau actual antes de pivotar
            steps.append(self.tableau.copy())
            pivot = self._find_pivot()
            if pivot is None:
                break
            pivot_row, pivot_col = pivot
            self._pivot(pivot_row, pivot_col)

        # Extraer la solución incluyendo variables de decisión y holgura
        solution = np.zeros(self.num_vars + self.num_constraints)  # Considerar todas las variables

        for i in range(self.num_vars + self.num_constraints):  # Incluir variables de holgura
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col) == 1:
                solution[i] = self.tableau[np.where(col == 1)[0][0], -1]

        optimal_value = self.tableau[-1, -1]

        # Ajustar el valor óptimo para minimización
        if self.problem_type == 'min':
            optimal_value = -optimal_value

        # Crear nombres de variables (incluyendo holgura)
        variable_names = [f"x{i + 1}" for i in range(self.num_vars)] + \
                         [f"s{j + 1}" for j in range(self.num_constraints)] + ["RHS"]

        return solution, optimal_value, steps, variable_names

    def get_c_ranges(self):
        # Implementa la lógica para obtener los rangos de los coeficientes de la función objetivo
        return {"min": [0] * self.num_vars, "max": [100] * self.num_vars}  # Ejemplo

    def get_b_ranges(self):
        # Implementa la lógica para obtener los rangos de los términos independientes
        return {"min": [0] * self.num_constraints, "max": [100] * self.num_constraints}

################################ METODO GRAN M #####################################
@app.route('/bigm', methods=['POST'])
def bigm_solver():
    try:
        data = request.json

        # Verificar que los datos requeridos estén presentes
        required_fields = ['c', 'A', 'b', 'constraint_types']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Falta el campo requerido: {field}"}), 400

        c = np.array(data['c'], dtype=float)
        A = np.array(data['A'], dtype=float)
        b = np.array(data['b'], dtype=float)
        constraint_types = data['constraint_types']
        problem_type = data.get('problem_type', 'max')
        M = data.get('M', 1e6)
        problem_description = data.get('problem_description', '')

        # Validar que las dimensiones de A, b y constraint_types coincidan
        m, n = A.shape
        if len(b) != m or len(constraint_types) != m:
            return jsonify({"error": "Las dimensiones de A, b y constraint_types no coinciden."}), 400

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia de BigM
        bigm = BigMWithSteps(c, A, b, constraint_types, problem_type, M)
        solution, optimal_value, steps, variable_names = bigm.solve_with_steps()

        # Verificar si la solución es válida
        if solution is None:
            return jsonify({"error": "El problema no tiene solución factible."}), 400

        # Verificar si hay iteraciones
        if steps is None:
            return jsonify({"error": "No se generaron iteraciones en el método Big M."}), 400

        # Generar el análisis de sensibilidad
        sensitivity_analysis = {
            "c_ranges": bigm.get_c_ranges(),
            "b_ranges": bigm.get_b_ranges()
        }

        # Crear el mensaje para Gemini
        gemini_prompt = f"""
        Contexto del problema: {problem_description}
        Resultados del cálculo:
        - Solución óptima: {solution}
        - Valor óptimo: {optimal_value}
        - Análisis de sensibilidad: {sensitivity_analysis}
        Por favor, proporciona un análisis de sensibilidad basado en los resultados anteriores basado en el contexto del problema.
        """
        # Llamar a Gemini
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt,
        )
        interpretation = gemini_response.text
        print(interpretation)

        # Devolver la solución, el valor óptimo, las tablas intermedias, los nombres de las variables y la interpretación
        return jsonify({
            "solution": solution.tolist() if solution is not None else [],
            "optimal_value": optimal_value,
            "steps": [step.tolist() for step in steps] if steps is not None else [],
            "variable_names": variable_names,
            "sensitivity_analysis": sensitivity_analysis,
            "interpretation": interpretation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

################################ METODO 2 FASES #####################################
@app.route('/two-phase', methods=['POST'])
def two_phase_solver():
    try:
        data = request.json
        print("📌 JSON recibido:", json.dumps(data, indent=2))  # Mejor formato para debugging

        # Validar datos
        required_fields = ["c", "A", "b", "constraint_types", "problem_type"]
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            return jsonify({
                "error": f"Faltan parámetros en la solicitud: {', '.join(missing_fields)}"
            }), 400

        # Convertir datos a arrays numpy
        try:
            c = np.array(data['c'], dtype=float)
            A = np.array(data['A'], dtype=float)
            b = np.array(data['b'], dtype=float)
            constraint_types = data['constraint_types']
            problem_type = data.get('problem_type', 'max')
            problem_description = data.get('problem_description', '')

            # Validación dimensional
            if A.shape[1] != len(c):
                return jsonify({
                    "error": f"Dimensiones incompatibles: A tiene {A.shape[1]} columnas pero c tiene {len(c)} elementos"
                }), 400
            if A.shape[0] != len(b):
                return jsonify({
                    "error": f"Dimensiones incompatibles: A tiene {A.shape[0]} filas pero b tiene {len(b)} elementos"
                }), 400
            if len(constraint_types) != len(b):
                return jsonify({
                    "error": f"Número incorrecto de tipos de restricciones: se esperaban {len(b)}, se recibieron {len(constraint_types)}"
                }), 400

        except ValueError as e:
            return jsonify({
                "error": f"Error al convertir los datos: {str(e)}"
            }), 400

        # Crear y resolver con el solver
        solver = TwoPhaseWithSteps(c, A, b, constraint_types, problem_type)
        solution, optimal_value, steps, variable_names = solver.solve_with_steps()

        # Verificar si se encontró solución
        if solution is None:
            return jsonify({
                "error": "No se encontró una solución óptima para el problema."
            }), 400

        # Obtener análisis de sensibilidad
        sensitivity_analysis = solver.get_sensitivity_analysis()
        # Crear el mensaje para Gemini
        gemini_prompt = f"""
        Contexto del problema: {problem_description}
        Resultados del cálculo:
        - Solución óptima: {solution}
        - Valor óptimo: {optimal_value}
        - Análisis de sensibilidad: {sensitivity_analysis}
        Por favor, proporciona un análisis de sensibilidad basado en los resultados anteriores basado en el contexto del problema.
        """

        # Llamar a Gemini
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt,
        )
        interpretation = gemini_response.text
        print(interpretation)  # Imprime la interpretación en la consola para depuración

        # Preparar respuesta
        response = {
            "solution": solution.tolist(),
            "optimal_value": float(optimal_value),
            "steps": [step.tolist() for step in steps],
            "variable_names": variable_names,
            "sensitivity_analysis": sensitivity_analysis,
            "interpretation": interpretation,
            "status": "success"
        }

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # Log del error
        return jsonify({
            "error": f"Error en el procesamiento: {str(e)}",
            "status": "error"
        }), 500



################################ METODO DUAL #####################################
@app.route('/dual', methods=['POST'])
def dual_solver():
    try:
        data = request.json
        c = np.array(data['c'], dtype=float)  # Convertir a float
        A = np.array(data['A'], dtype=float)  # Convertir a float
        b = np.array(data['b'], dtype=float)  # Convertir a float
        problem_type = data.get('problem_type', 'max')
        problem_description = data.get('problem_description', '')

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia de DualSimplex
        dual_simplex = DualSimplex(c, A, b, problem_type)
        solution, optimal_value, steps, variable_names = dual_simplex.solve_with_steps()

        # Obtener las variables X del primal (precios sombra del dual)
        primal_solution = {}
        for i, var_name in enumerate(variable_names):
            if var_name.startswith("Y"):  # Las variables Y del dual son los precios sombra del primal
                primal_solution[f"X{i+1}"] = solution[var_name]  # Asignar los valores de Y a X

        # Generar el análisis de sensibilidad
        sensitivity_analysis = {
            "c_ranges": dual_simplex.get_c_ranges(),
            "b_ranges": dual_simplex.get_b_ranges()
        }

        # Crear el mensaje para Gemini
        gemini_prompt = f"""
        Contexto del problema: {problem_description}
        Resultados del cálculo:
        - Solución óptima (Dual): {solution}
        - Solución óptima (Primal): {primal_solution}
        - Valor óptimo: {optimal_value}
        - Análisis de sensibilidad: {sensitivity_analysis}
        Por favor, proporciona un análisis de sensibilidad basado en los resultados anteriores basado en el contexto del problema.
        """
        # Llamar a Gemini
        gemini_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=gemini_prompt,
        )
        interpretation = gemini_response.text
        print(interpretation)

        # Devolver la solución, el valor óptimo, las tablas intermedias, los nombres de las variables y la interpretación
        return jsonify({
            "solution": solution,  # Solución del dual
            "primal_solution": primal_solution,  # Solución del primal
            "optimal_value": optimal_value,  # Valor óptimo
            "steps": steps,  # Tablas intermedias
            "variable_names": variable_names,  # Nombres de las variables
            "sensitivity_analysis": sensitivity_analysis,  # Análisis de sensibilidad
            "interpretation": interpretation  # Interpretación de Gemini
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ TRANSPORTE: ESQUINA NOROESTE #####################################
@app.route('/esquinanoroeste', methods=['POST'])
def calcular_esquina_noroeste():
    # Recibir datos del frontend
    data = request.get_json()

    costos = data.get('costos')
    suministros = data.get('suministros')
    demandas = data.get('demandas')

    # Validar los datos de entrada
    if not costos or not suministros or not demandas:
        return jsonify({"error": "Faltan datos necesarios"}), 400

    try:
        # Llamar al método para calcular la solución
        resultado = obtener_resultado(costos, suministros, demandas)

        # Devolver el resultado en formato JSON
        return jsonify({"resultado": resultado})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ TRANSPORTE: COSTO MINIMO #####################################
@app.route('/costominimo', methods=['POST'])
def calcular():
    data = request.get_json()

    # Obtener los datos del cuerpo de la solicitud
    costos = data['costos']
    suministros = data['suministros']
    demandas = data['demandas']

    # Llamar a la función para calcular el costo mínimo
    resultado = calcular_costo_minimo(costos, suministros, demandas)

    # Devolver la respuesta con los resultados
    return jsonify({'resultado': resultado})

@app.route('/costo-minimo', methods=['POST'])
def calcular1():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()
        costos = data['costos']
        suministros = data['suministros']
        demandas = data['demandas']

        # Validar que los datos no estén vacíos
        if not costos or not suministros or not demandas:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        # Llamar a la función para calcular el costo mínimo
        resultado = calcular_costominimo(costos, suministros, demandas)

        # Devolver la respuesta con los resultados
        return jsonify({'resultado': resultado})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ TRANSPORTE: VOGEL #####################################
@app.route('/vogel', methods=['POST'])
def calcular_vogel():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()
        costos = data['costos']
        suministros = data['suministros']
        demandas = data['demandas']

        # Validar que los datos no estén vacíos
        if not costos or not suministros or not demandas:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        # Llamar a la función para calcular el método de Vogel
        resultado, penalizaciones = metodo_vogel(costos, suministros, demandas)

        # Devolver la respuesta con los resultados y las penalizaciones
        return jsonify({
            'resultado': resultado,
            'penalizaciones': penalizaciones
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ REDES: CAMINO MAS CORTO #####################################
@app.route('/caminocorto', methods=['POST'])
def shortest_path():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()
        graph = data['graph']  # Grafo representado como un diccionario
        start = data['start']  # Nodo inicial
        end = data['end']      # Nodo final

        # Validar que los datos no estén vacíos
        if not graph or not start or not end:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        # Llamar a la función para calcular el camino más corto
        path, distance = dijkstra(graph, start, end)

        # Devolver la respuesta con el camino y la distancia
        return jsonify({
            'path': path,
            'distance': distance
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ REDES: FLUJO MAXIMO #####################################
@app.route('/flujomaximo', methods=['POST'])
def flujo_maximo():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()
        graph = data['graph']  # Grafo representado como un diccionario
        source = data['source']  # Nodo fuente
        sink = data['sink']      # Nodo sumidero

        # Validar que los datos no estén vacíos
        if not graph or not source or not sink:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        # Llamar a la función para calcular el flujo máximo
        max_flow, used_edges = edmonds_karp(graph, source, sink)

        # Devolver la respuesta con el flujo máximo y las aristas utilizadas
        return jsonify({
            'max_flow': max_flow,
            'used_edges': used_edges
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ REDES: ARBOL EXPANSION MINIMA #####################################
@app.route('/mst', methods=['POST'])
def mst():
    try:
        # Obtener los datos del cuerpo de la solicitud
        data = request.get_json()
        edges = data['edges']  # Lista de aristas con pesos
        nodes = data['nodes']  # Lista de nodos

        # Validar que los datos no estén vacíos
        if not edges or not nodes:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        # Llamar a la función para calcular el MST
        total_cost, mst_edges = kruskal(edges, nodes)

        # Devolver la respuesta con el costo total y las aristas del MST
        return jsonify({
            'total_cost': total_cost,
            'mst_edges': mst_edges
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)