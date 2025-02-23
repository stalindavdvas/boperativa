import numpy as np

class DualSimplex:
    def __init__(self, c, A, b, problem_type='max'):
        self.c = np.array(c, dtype=float)  # Convertir a float
        self.A = np.array(A, dtype=float)  # Convertir a float
        self.b = np.array(b, dtype=float)  # Convertir a float
        self.problem_type = problem_type
        self.num_vars = len(c)
        self.num_constraints = len(b)
        self.tableau = self._create_initial_tableau()

    def _create_initial_tableau(self):
        dual_A = self.A.T
        dual_c = self.b
        dual_b = self.c

        tableau = np.zeros((self.num_vars + 1, self.num_constraints + self.num_vars + 1), dtype=float)
        tableau[:self.num_vars, :self.num_constraints] = dual_A
        for i in range(self.num_vars):
            tableau[i, self.num_constraints + i] = 1
        tableau[:self.num_vars, -1] = dual_b
        tableau[-1, :self.num_constraints] = -np.array(dual_c)
        return tableau

    def solve_with_steps(self):
        steps = []
        variable_names = [f"Y{i+1}" for i in range(self.num_constraints)] + \
                         [f"S{j+1}" for j in range(self.num_vars)] + ["RHS"]
        base_variables = [f"S{j+1}" for j in range(self.num_vars)] + ["W"]

        while True:
            # Guardar la tabla actual para las iteraciones
            tabla_con_nombres = {
                "base": base_variables.copy(),
                "columnas": variable_names,
                "valores": [[float(val) for val in fila] for fila in self.tableau.tolist()],  # Convertir a float
            }
            steps.append(tabla_con_nombres)

            # Condición de optimalidad
            if all(val >= 0 for val in self.tableau[-1, :-1]):
                break

            # Elegir columna pivote (variable entrante)
            pivot_col = int(np.argmin(self.tableau[-1, :-1]))  # Convertir a int
            variable_entrante = variable_names[pivot_col]

            # Elegir fila pivote (variable saliente)
            ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
            ratios[self.tableau[:-1, pivot_col] <= 0] = np.inf
            pivot_row = int(np.argmin(ratios))  # Convertir a int
            variable_saliente = base_variables[pivot_row]

            # Actualizar variables básicas
            base_variables[pivot_row] = variable_entrante

            # Pivoteo
            pivot_value = float(self.tableau[pivot_row, pivot_col])  # Convertir a float
            self.tableau[pivot_row, :] /= pivot_value
            for i in range(self.tableau.shape[0]):
                if i != pivot_row:
                    self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

        # Extraer solución óptima
        solution = {}
        for i, var in enumerate(variable_names[:-1]):  # Excluimos RHS
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == len(col) - 1:
                row = int(np.where(col == 1)[0][0])  # Convertir a int
                solution[var] = float(self.tableau[row, -1])  # Convertir a float
            else:
                solution[var] = 0.0  # Variables no básicas

        optimal_value = float(self.tableau[-1, -1])  # Convertir a float

        # Si el problema original era de minimización, ajustamos el valor óptimo
        if self.problem_type == 'min':
            optimal_value = -optimal_value

        return solution, optimal_value, steps, variable_names

    def get_c_ranges(self):
        """
        Calcula los rangos de variación para los coeficientes de la función objetivo.
        """
        c_ranges = {}
        for i in range(self.num_vars):
            c_ranges[f"c{i+1}"] = {
                "lower_bound": float(self.c[i] - 1),  # Convertir a float
                "upper_bound": float(self.c[i] + 1)   # Convertir a float
            }
        return c_ranges

    def get_b_ranges(self):
        """
        Calcula los rangos de variación para los valores del lado derecho de las restricciones.
        """
        b_ranges = {}
        for i in range(self.num_constraints):
            b_ranges[f"b{i+1}"] = {
                "lower_bound": float(self.b[i] - 1),  # Convertir a float
                "upper_bound": float(self.b[i] + 1)   # Convertir a float
            }
        return b_ranges