import numpy as np

class DualSimplex:
    def __init__(self, c, A, b, problem_type='max'):
        """
        Inicializa el problema de programación lineal para el método dual.
        :param c: Coeficientes de la función objetivo del problema primal (1D array).
        :param A: Coeficientes de las restricciones del problema primal (2D array).
        :param b: Lado derecho de las restricciones del problema primal (1D array).
        :param problem_type: 'max' para maximización, 'min' para minimización.
        """
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.problem_type = problem_type
        self.num_vars = len(c)
        self.num_constraints = len(b)

        # Validar que las dimensiones sean consistentes
        if self.A.shape[0] != self.num_constraints or self.A.shape[1] != self.num_vars:
            raise ValueError("Las dimensiones de A no coinciden con el número de restricciones o variables.")
        if len(self.b) != self.num_constraints:
            raise ValueError("El tamaño de b no coincide con el número de restricciones.")

        # Convertir minimización en maximización
        if self.problem_type == 'min':
            self.c = -self.c  # Multiplicamos por -1 para convertir minimización en maximización

        # Crear el tableau inicial del problema dual
        self.tableau = self._create_initial_tableau()

    def _create_initial_tableau(self):
        """
        Crea el tableau inicial para el problema dual.
        """
        # Transponer la matriz A para obtener el problema dual
        dual_A = self.A.T
        dual_c = self.b
        dual_b = self.c

        # Construir la matriz ampliada
        tableau = np.zeros((self.num_vars + 1, self.num_constraints + self.num_vars + 1))

        # Rellenar la matriz A transpuesta
        tableau[:self.num_vars, :self.num_constraints] = dual_A

        # Rellenar las variables de holgura
        for i in range(self.num_vars):
            tableau[i, self.num_constraints + i] = 1

        # Rellenar el lado derecho (coeficientes de la función objetivo del primal)
        tableau[:self.num_vars, -1] = dual_b

        # Función objetivo del dual (lado derecho del primal)
        tableau[-1, :self.num_constraints] = -np.array(dual_c)

        return tableau

    def solve_with_steps(self):
        """
        Resuelve el problema dual usando el método simplex y captura las tablas intermedias.
        """
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
            pivot_col = np.argmin(self.tableau[-1, :-1])
            variable_entrante = variable_names[pivot_col]

            # Elegir fila pivote (variable saliente)
            ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
            ratios[self.tableau[:-1, pivot_col] <= 0] = np.inf
            pivot_row = np.argmin(ratios)
            variable_saliente = base_variables[pivot_row]

            # Actualizar variables básicas
            base_variables[pivot_row] = variable_entrante

            # Pivoteo
            pivot_value = self.tableau[pivot_row, pivot_col]
            self.tableau[pivot_row, :] /= pivot_value
            for i in range(self.tableau.shape[0]):
                if i != pivot_row:
                    self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

        # Extraer solución óptima
        solution = {}
        for i, var in enumerate(variable_names[:-1]):  # Excluimos RHS
            col = self.tableau[:, i]
            if np.sum(col == 1) == 1 and np.sum(col == 0) == len(col) - 1:
                row = np.where(col == 1)[0][0]
                solution[var] = float(self.tableau[row, -1])  # Convertir a float
            else:
                solution[var] = 0.0  # Variables no básicas

        optimal_value = float(self.tableau[-1, -1])  # Convertir a float

        # Si el problema original era de minimización, ajustamos el valor óptimo
        if self.problem_type == 'min':
            optimal_value = -optimal_value

        return solution, optimal_value, steps, variable_names