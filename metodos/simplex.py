import numpy as np


class Simplex:
    def __init__(self, c, A, b, problem_type='max'):
        """
        Inicializa el problema de programación lineal.

        :param c: Coeficientes de la función objetivo (1D array).
        :param A: Coeficientes de las restricciones (2D array).
        :param b: Lado derecho de las restricciones (1D array).
        :param problem_type: 'max' para maximización, 'min' para minimización.
        """
        self.c = c
        self.A = A
        self.b = b
        self.problem_type = problem_type
        self.num_vars = len(c)
        self.num_constraints = len(b)

        # Convertir minimización en maximización
        if self.problem_type == 'min':
            self.c = -self.c  # Convertir la función objetivo

        # Crear el tableau inicial
        self.tableau = self._create_initial_tableau()

    def _create_initial_tableau(self):
        """
        Crea el tableau inicial para el método simplex.
        """
        # Añadir variables de holgura
        slack_vars = np.eye(self.num_constraints)
        tableau = np.hstack((self.A, slack_vars))

        # Añadir la función objetivo (con signo negativo para maximización)
        c_extended = np.hstack((self.c, np.zeros(self.num_constraints)))
        tableau = np.vstack((tableau, -c_extended))

        # Añadir el lado derecho de las restricciones
        b_extended = np.hstack((self.b, [0]))
        tableau = np.column_stack((tableau, b_extended))

        return tableau

    def _find_pivot(self):
        """
        Encuentra el pivote para la iteración actual.
        """
        # Encontrar la columna pivote (la más negativa en la última fila)
        last_row = self.tableau[-1, :-1]
        pivot_col = np.argmin(last_row)

        if last_row[pivot_col] >= 0:
            return None  # Solución óptima encontrada

        # Encontrar la fila pivote usando la regla del cociente mínimo
        ratios = self.tableau[:-1, -1] / self.tableau[:-1, pivot_col]
        ratios[ratios < 0] = np.inf  # Ignorar ratios negativos
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
        Resuelve el problema de programación lineal usando el método simplex.
        """
        while True:
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

        return solution, optimal_value

