import numpy as np

class GranM:
    def __init__(self, c, A, b, constraint_types, problem_type='max', M=1e6):
        """
        Inicializa el problema de programación lineal con el método de la Gran M.
        :param c: Coeficientes de la función objetivo (1D array).
        :param A: Coeficientes de las restricciones (2D array).
        :param b: Lado derecho de las restricciones (1D array).
        :param constraint_types: Lista de tipos de restricciones ('<=', '>=', '=').
        :param problem_type: 'max' para maximización, 'min' para minimización.
        :param M: Valor grande para penalizar las variables artificiales.
        """
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)
        self.constraint_types = constraint_types
        self.problem_type = problem_type
        self.M = M
        self.num_vars = len(c)
        self.num_constraints = len(b)
        # Convertir minimización en maximización
        if self.problem_type == 'min':
            self.c = -self.c
        # Crear el tableau inicial
        self.tableau = self._create_initial_tableau()

    def _create_initial_tableau(self):
        """
        Crea el tableau inicial para el método de la Gran M.
        """
        # Determinar el número de variables de holgura y artificiales
        slack_vars = [1 if ct in ['<=', '>='] else 0 for ct in self.constraint_types]
        artificial_vars = [1 if ct in ['>=', '='] else 0 for ct in self.constraint_types]
        num_slack = sum(slack_vars)
        num_artificial = sum(artificial_vars)

        # Construir la matriz ampliada
        tableau = np.zeros((self.num_constraints + 1, self.num_vars + num_slack + num_artificial + 1))

        # Rellenar la matriz A
        tableau[:self.num_constraints, :self.num_vars] = self.A

        # Rellenar las variables de holgura
        slack_index = self.num_vars
        for i, ct in enumerate(self.constraint_types):
            if ct == '<=':
                tableau[i, slack_index] = 1
                slack_index += 1
            elif ct == '>=':
                tableau[i, slack_index] = -1
                slack_index += 1

        # Rellenar las variables artificiales
        artificial_index = self.num_vars + num_slack
        for i, ct in enumerate(self.constraint_types):
            if ct in ['>=', '=']:
                tableau[i, artificial_index] = 1
                artificial_index += 1

        # Rellenar el lado derecho
        tableau[:self.num_constraints, -1] = self.b

        # Rellenar la función objetivo
        c_extended = np.hstack((self.c, np.zeros(num_slack), np.full(num_artificial, -self.M)))
        tableau[-1, :-1] = -c_extended

        return tableau

    def _find_pivot(self):
        """
        Encuentra el pivote para la iteración actual.
        """
        last_row = self.tableau[-1, :-1]
        pivot_col = np.argmin(last_row)
        if last_row[pivot_col] >= 0:
            return None  # Solución óptima encontrada
        ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
        ratios[ratios < 0] = np.inf
        pivot_row = np.argmin(ratios)
        if ratios[pivot_row] == np.inf:
            raise ValueError("El problema es no acotado.")
        return pivot_row, pivot_col

    def _pivot(self, pivot_row, pivot_col):
        """
        Realiza el pivoteo en el tableau.
        """
        pivot_element = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_element
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                self.tableau[i, :] -= self.tableau[i, pivot_col] * self.tableau[pivot_row, :]

    def solve(self):
        """
        Resuelve el problema de programación lineal usando el método de la Gran M.
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
        return solution, optimal_value, steps