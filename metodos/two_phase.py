import numpy as np
from typing import List, Tuple, Dict, Optional
from pulp import *


class TwoPhaseWithSteps:
    def __init__(self, c, A, b, constraint_types, problem_type='max'):
        self.c = c
        self.A = A
        self.b = b
        self.constraint_types = constraint_types
        self.problem_type = problem_type
        self.m, self.n = A.shape
        self.steps = []
        self.variable_names = []
        self._prepare_problem()

    def _prepare_problem(self):
        self._standardize_problem()
        self.original_vars = [f'x{i + 1}' for i in range(self.n)]
        self.slack_vars = [f's{i + 1}' for i in range(self.m)]
        self.artificial_vars = [f'a{i + 1}' for i in range(self.m)]
        self.variable_names = self.original_vars + self.slack_vars + self.artificial_vars

    def _standardize_problem(self):
        if self.problem_type == 'min':
            self.c = -self.c
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.b[i] = -self.b[i]
                self.A[i] = -self.A[i]
                if self.constraint_types[i] != '=':
                    self.constraint_types[i] = '<=' if self.constraint_types[i] == '>=' else '>='

    def _get_current_tableau(self, prob):
        """
        Obtiene el tableau actual del problema corregido para PuLP.
        """
        tableau = []
        all_vars = prob.variables()

        # Construir diccionario de índices de variables
        var_indices = {var.name: i for i, var in enumerate(all_vars)}

        # Fila de la función objetivo
        obj_row = [0.0] * (len(all_vars) + 1)
        for var, coef in prob.objective.items():
            if var.name in var_indices:
                obj_row[var_indices[var.name]] = coef
        tableau.append(obj_row)

        # Filas de restricciones
        for name, constraint in prob.constraints.items():
            row = [0.0] * (len(all_vars) + 1)
            for var, coef in constraint.items():
                if var.name in var_indices:
                    row[var_indices[var.name]] = coef
            row[-1] = -constraint.constant
            tableau.append(row)

        return np.array(tableau)

    def solve_with_steps(self):
        # Fase 1
        prob_phase1 = LpProblem("Phase1", LpMinimize)
        vars_dict = {}

        # Crear variables
        for var_name in self.variable_names:
            vars_dict[var_name] = LpVariable(var_name, lowBound=0)

        # Función objetivo Fase 1
        prob_phase1 += lpSum(vars_dict[v] for v in self.artificial_vars)

        # Restricciones Fase 1
        for i in range(self.m):
            constraint_expr = lpSum(self.A[i][j] * vars_dict[self.original_vars[j]]
                                    for j in range(self.n))
            if self.constraint_types[i] == '<=':
                constraint_expr += vars_dict[self.slack_vars[i]]
            elif self.constraint_types[i] == '>=':
                constraint_expr -= vars_dict[self.slack_vars[i]]
            constraint_expr += vars_dict[self.artificial_vars[i]]
            prob_phase1 += constraint_expr == self.b[i]

        # Resolver Fase 1
        prob_phase1.solve(PULP_CBC_CMD(msg=False))
        self.steps.append(self._get_current_tableau(prob_phase1))

        if prob_phase1.status != LpStatusOptimal or value(prob_phase1.objective) > 1e-10:
            return None, None, self.steps, self.variable_names

        # Fase 2
        prob_phase2 = LpProblem("Phase2",
                                LpMaximize if self.problem_type == 'max' else LpMinimize)

        # Función objetivo original
        prob_phase2 += lpSum(self.c[j] * vars_dict[self.original_vars[j]]
                             for j in range(self.n))

        # Restricciones Fase 2
        for i in range(self.m):
            constraint_expr = lpSum(self.A[i][j] * vars_dict[self.original_vars[j]]
                                    for j in range(self.n))
            if self.constraint_types[i] == '<=':
                constraint_expr += vars_dict[self.slack_vars[i]]
            elif self.constraint_types[i] == '>=':
                constraint_expr -= vars_dict[self.slack_vars[i]]
            prob_phase2 += constraint_expr == self.b[i]

        # Resolver Fase 2
        prob_phase2.solve(PULP_CBC_CMD(msg=False))
        self.steps.append(self._get_current_tableau(prob_phase2))

        # Extraer solución
        solution = np.zeros(self.n)
        for j in range(self.n):
            solution[j] = value(vars_dict[self.original_vars[j]])

        optimal_value = value(prob_phase2.objective)
        if self.problem_type == 'min':
            optimal_value = -optimal_value

        return solution, optimal_value, self.steps, self.variable_names

    def get_sensitivity_analysis(self):
        sensitivity = {
            'objective_coefficients': {},
            'rhs_values': {},
            'decision_variables': {}
        }

        # Análisis de coeficientes de la función objetivo
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
                sensitivity['objective_coefficients'][f'x{j + 1}'] = {
                    'current_value': float(c_original),
                    'upper_bound': float(c_original + delta),
                    'lower_bound': float(c_original - delta),
                    'impact': float((val_up - val_down) / (2 * delta))
                }

        # Análisis de términos independientes
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
                sensitivity['rhs_values'][f'constraint_{i + 1}'] = {
                    'current_value': float(b_original),
                    'upper_bound': float(b_original + delta),
                    'lower_bound': float(b_original - delta),
                    'shadow_price': float((val_up - val_down) / (2 * delta))
                }

        return sensitivity