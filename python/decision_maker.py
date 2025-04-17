"""Implementation of Decision Maker with UTA decision function."""

import matplotlib.pyplot as plt
import numpy as np


class DecisionMaker:

    def __init__(self, n_criteria, n_pieces):
        self.n_criteria = n_criteria
        self.n_pieces = n_pieces

        self.slopes = self.build_random_decision_function()
        self.breakpoints_x = np.linspace(0, self.n_pieces, self.n_pieces+1)
        self.break_point_y = np.stack([[0] + [np.sum(self.slopes[i][:j+1]) for j in range(self.n_pieces)] for i in range(self.n_criteria)])

    def build_random_decision_function(self):
        slopes = []
        for i in range(self.n_criteria):
            crit_slopes = []
            for j in range(self.n_pieces):
                if i == 0 and j == 0:
                    crit_slopes.append(1.)
                else:
                    crit_slopes.append(np.random.uniform(0, 10))
            slopes.append(crit_slopes)
        return np.stack(slopes)

    def plot_decision_function(self):
        x = np.linspace(0, self.n_pieces, self.n_pieces+1)
        for i in range(self.n_criteria):
            y = [0]
            for j in range(self.n_pieces):
                y.append(y[-1] + self.slopes[i][j])
            plt.plot(x, y, label=f'Criterion {i+1}')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Utility")
        plt.show()

    def get_ui(self, criterion, value):
        marginal_utility = 0
        for i in range(self.n_pieces):
            sub_value = np.max([np.min([1., value - i]), 0.]) * self.slopes[criterion][i]
            marginal_utility += sub_value
        return marginal_utility

    def answer(self, criterion_i, criterion_j, q_i, p_i, q_j):
        du = self.get_ui(criterion_i, p_i) - self.get_ui(criterion_i, q_i)
        u_j = self.get_ui(criterion_j, q_j)

        if du > 0:
            if du > u_j:
                return None
            else:
                for i in range(self.n_pieces):
                    if u_j - self.break_point_y[criterion_j][i] >= du and u_j - self.break_point_y[criterion_j][i+1] < du:
                        
                        dv = du - u_j + self.break_point_y[criterion_j][i+1]
                        return i+1 - (dv / self.slopes[criterion_j][i])
        else:
            if -du > (self.break_point_y[criterion_j][-1] - u_j):
                return None
            else:
                for i in range(self.n_pieces):
                    if self.break_point_y[criterion_j][i+1] - u_j > -du and self.break_point_y[criterion_j][i] - u_j <= -du:
                        
                        dv = -du + u_j - self.break_point_y[criterion_j][i]
                        return i + dv / self.slopes[criterion_j][i]
        

class HiddenDecisionMakers:

    def __init__(self, n_dms, n_criteria, n_pieces):
        self.dms = [DecisionMaker(n_criteria, n_pieces) for i in range(n_dms)]

    def answer(self, criterion_i, criterion_j, q_i, p_i, q_j):
        answers = [dm.answer(criterion_i, criterion_j, q_i, p_i, q_j) for dm in self.dms]
        return np.random.permutation(answers)
