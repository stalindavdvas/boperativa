import numpy as np
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize, LpStatus, value, PULP_CBC_CMD
class BigMWithSteps:
    def __init__(self, c, A, b, constraint_types, problem_type='max', M=1e6):
        """
        Inicializa el método de la Gran M.

        Args:
            c: Vector de coeficientes de la función objetivo.
            A: Matriz de coeficientes de las restricciones.
            b: Vector de términos independientes.
            constraint_types: Lista de tipos de restricciones ('<=', '>=', '=').
            problem_type: 'max' o 'min'.
            M: Valor para el método de la Gran M.
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.constraint_types = constraint_types
        self.problem_type = problem_type
        self.M = M if problem_type == 'max' else -M
        self.m, self.n = A.shape
        self.steps = []
        self.variable_names = []
        self.prob = None
        self._prepare_problem()

    def _prepare_problem(self):
        """Prepara el problema para el método de la Gran M."""
        # Estandarizar el problema
        self._standardize_problem()

        # Crear nombres de variables
        self.original_vars = [f'x{i + 1}' for i in range(self.n)]
        self.slack_vars = [f's{i + 1}' for i in range(self.m)]
        self.artificial_vars = [f'a{i + 1}' for i in range(self.m)]
        self.variable_names = self.original_vars + self.slack_vars + self.artificial_vars

    def _standardize_problem(self):
        """Estandariza el problema convirtiendo las restricciones."""
        if self.problem_type == 'min':
            self.c = -self.c

        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.A[i] = -self.A[i]
                if self.constraint_types[i] != '=':
                    self.constraint_types[i] = '<=' if self.constraint_types[i] == '>=' else '>='

    def solve_with_steps(self):
        """
        Resuelve el problema usando el método de la Gran M.

        Returns:
            tuple: (solución, valor óptimo, pasos, nombres de variables)
        """
        try:
            # Crear el problema
            self.prob = LpProblem("BigM_Method", LpMaximize if self.problem_type == 'max' else LpMinimize)

            # Crear variables
            vars_dict = {}
            for var_name in self.original_vars:
                vars_dict[var_name] = LpVariable(var_name, lowBound=0)

            # Agregar variables de holgura y artificiales
            for i in range(self.m):
                if self.constraint_types[i] in ['<=', '>=']:
                    vars_dict[self.slack_vars[i]] = LpVariable(self.slack_vars[i], lowBound=0)
                if self.constraint_types[i] in ['>=', '=']:
                    vars_dict[self.artificial_vars[i]] = LpVariable(self.artificial_vars[i], lowBound=0)

            # Función objetivo con términos de penalización M
            objective = lpSum(self.c[j] * vars_dict[self.original_vars[j]] for j in range(self.n))
            for i in range(self.m):
                if self.constraint_types[i] in ['>=', '=']:
                    objective += self.M * vars_dict[self.artificial_vars[i]]
            self.prob += objective

            # Restricciones
            for i in range(self.m):
                constraint_expr = lpSum(self.A[i][j] * vars_dict[self.original_vars[j]] for j in range(self.n))
                if self.constraint_types[i] == '<=':
                    constraint_expr += vars_dict[self.slack_vars[i]]
                elif self.constraint_types[i] == '>=':
                    constraint_expr -= vars_dict[self.slack_vars[i]]
                    constraint_expr += vars_dict[self.artificial_vars[i]]
                else:  # '='
                    constraint_expr += vars_dict[self.artificial_vars[i]]
                self.prob += constraint_expr == self.b[i]

            # Resolver
            self.prob.solve(PULP_CBC_CMD(msg=False))

            # Verificar si se encontró solución óptima
            if self.prob.status != LpStatus[1]:  # 1 = Optimal
                return None, None, None, None

            # Extraer solución
            solution = np.zeros(self.n)
            for j in range(self.n):
                solution[j] = value(vars_dict[self.original_vars[j]])

            optimal_value = value(self.prob.objective)
            if self.problem_type == 'min':
                optimal_value = -optimal_value

            return solution, optimal_value, self.steps, self.variable_names
        except Exception as e:
            print(f"Error en solve_with_steps: {e}")
            return None, None, None, None

    def get_c_ranges(self):
        """Obtiene los rangos de sensibilidad para los coeficientes de la función objetivo."""
        sensitivity = {}
        for j in range(self.n):
            c_original = self.c[j]
            delta = max(abs(c_original) * 0.01, 1.0)

            # Análisis de incremento
            self.c[j] = c_original + delta
            _, val_up, _, _ = self.solve_with_steps()

            # Análisis de decremento
            self.c[j] = c_original - delta
            _, val_down, _, _ = self.solve_with_steps()

            # Restaurar valor original
            self.c[j] = c_original

            if val_up is not None and val_down is not None:
                sensitivity[f'x{j + 1}'] = {
                    'current_value': float(c_original),
                    'upper_bound': float(c_original + delta),
                    'lower_bound': float(c_original - delta),
                    'impact': float((val_up - val_down) / (2 * delta))
                }
        return sensitivity

    def get_b_ranges(self):
        """Obtiene los rangos de sensibilidad para los términos independientes."""
        sensitivity = {}
        for i in range(len(self.b)):
            b_original = self.b[i]
            delta = max(abs(b_original) * 0.01, 1.0)

            # Análisis de incremento
            self.b[i] = b_original + delta
            _, val_up, _, _ = self.solve_with_steps()

            # Análisis de decremento
            self.b[i] = b_original - delta
            _, val_down, _, _ = self.solve_with_steps()

            # Restaurar valor original
            self.b[i] = b_original

            if val_up is not None and val_down is not None:
                sensitivity[f'constraint_{i + 1}'] = {
                    'current_value': float(b_original),
                    'upper_bound': float(b_original + delta),
                    'lower_bound': float(b_original - delta),
                    'shadow_price': float((val_up - val_down) / (2 * delta))
                }
        return sensitivity