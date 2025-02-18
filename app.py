from flask import Flask, request, jsonify
import numpy as np

from metodos.EsquinaNoroeste import obtener_resultado
from metodos.arbolminimo import kruskal
from metodos.big_m import GranM
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

app = Flask(__name__)
CORS(app)
############### METODO SIMPLEX ##########################################
@app.route('/simplex', methods=['POST'])
def simplex_solver():
    try:
        data = request.json
        c = np.array(data['c'])
        A = np.array(data['A'])
        b = np.array(data['b'])
        problem_type = data.get('problem_type', 'max')

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia de Simplex
        simplex = SimplexWithSteps(c, A, b, problem_type)
        solution, optimal_value, steps, variable_names = simplex.solve_with_steps()

        # Devolver la solución, el valor óptimo, las tablas intermedias y los nombres de las variables
        return jsonify({
            "solution": solution.tolist(),
            "optimal_value": optimal_value,
            "steps": [step.tolist() for step in steps],  # Convertir arrays NumPy a listas
            "variable_names": variable_names
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

        # Extraer la solución
        solution = np.zeros(self.num_vars)
        for i in range(self.num_vars):
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col) == 1:
                solution[i] = self.tableau[np.where(col == 1)[0][0], -1]

        optimal_value = self.tableau[-1, -1]
        # Ajustar el valor óptimo para minimización
        if self.problem_type == 'min':
            optimal_value = -optimal_value

        # Crear nombres de variables
        variable_names = [f"x{i+1}" for i in range(self.num_vars)] + \
                         [f"s{j+1}" for j in range(self.num_constraints)] + ["RHS"]

        return solution, optimal_value, steps, variable_names
################################ METODO GRAN M #####################################
@app.route('/bigm', methods=['POST'])
def bigm_solver():
    try:
        data = request.json
        c = np.array(data['c'])
        A = np.array(data['A'])
        b = np.array(data['b'])
        constraint_types = data['constraint_types']
        problem_type = data.get('problem_type', 'max')
        M = data.get('M', 1e6)  # Valor grande para penalizar las variables artificiales

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia de BigM
        bigm = BigMWithSteps(c, A, b, constraint_types, problem_type, M)
        solution, optimal_value, steps, variable_names = bigm.solve_with_steps()

        # Devolver la solución, el valor óptimo, las tablas intermedias y los nombres de las variables
        return jsonify({
            "solution": solution.tolist(),
            "optimal_value": optimal_value,
            "steps": [step.tolist() for step in steps],
            "variable_names": variable_names
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


class BigMWithSteps(GranM):
    def solve_with_steps(self):
        """
        Resuelve el problema de programación lineal usando el método de la Gran M
        y captura las tablas intermedias.
        """
        steps = []
        while True:
            steps.append(self.tableau.copy())
            pivot = self._find_pivot()
            if pivot is None:
                break
            pivot_row, pivot_col = pivot
            self._pivot(pivot_row, pivot_col)

        # Extraer la solución
        solution = np.zeros(self.num_vars)
        for i in range(self.num_vars):
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col) == 1:
                solution[i] = self.tableau[np.where(col == 1)[0][0], -1]
        optimal_value = self.tableau[-1, -1]
        if self.problem_type == 'min':
            optimal_value = -optimal_value

        # Crear nombres de variables
        num_slack = sum([1 if ct in ['<=', '>='] else 0 for ct in self.constraint_types])
        num_artificial = sum([1 if ct in ['>=', '='] else 0 for ct in self.constraint_types])
        variable_names = [f"x{i+1}" for i in range(self.num_vars)] + \
                         [f"s{j+1}" for j in range(num_slack)] + \
                         [f"a{k+1}" for k in range(num_artificial)] + ["RHS"]
        return solution, optimal_value, steps, variable_names
################################ METODO 2 FASES #####################################
@app.route('/two-phase', methods=['POST'])
def two_phase_solver():
    try:
        data = request.json
        c = np.array(data['c'])
        A = np.array(data['A'])
        b = np.array(data['b'])
        constraint_types = data['constraint_types']
        problem_type = data.get('problem_type', 'max')

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia del método de las dos fases
        solver = TwoPhaseWithSteps(c, A, b, constraint_types, problem_type)
        solution, optimal_value, steps, variable_names = solver.solve_with_steps()

        # Devolver la solución, el valor óptimo, las tablas intermedias y los nombres de las variables
        return jsonify({
            "solution": solution.tolist(),
            "optimal_value": optimal_value,
            "steps": [step.tolist() for step in steps],
            "variable_names": variable_names
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
################################ METODO DUAL #####################################
@app.route('/dual', methods=['POST'])
def dual_solver():
    try:
        data = request.json
        c = np.array(data['c'])
        A = np.array(data['A'])
        b = np.array(data['b'])
        problem_type = data.get('problem_type', 'max')  # Por defecto, maximización

        # Validar que los datos no contengan NaN
        if np.any(np.isnan(c)) or np.any(np.isnan(A)) or np.any(np.isnan(b)):
            return jsonify({"error": "Los datos contienen valores inválidos (NaN)."}), 400

        # Crear una instancia del método dual
        solver = DualSimplex(c, A, b, problem_type=problem_type)
        solution, optimal_value, steps, variable_names = solver.solve_with_steps()

        # Devolver la solución, el valor óptimo, las tablas intermedias y los nombres de las variables
        return jsonify({
            "solution": solution,
            "optimal_value": optimal_value,
            "steps": steps,
            "variable_names": variable_names
        })
    except Exception as e:
        print(f"Error: {str(e)}")  # Imprime el error completo en la consola
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
################################ TRANSPORTE: ESQUINA NOROESTE #####################################

################################ TRANSPORTE: ESQUINA NOROESTE #####################################

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)