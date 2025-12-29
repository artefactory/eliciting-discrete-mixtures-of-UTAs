"""Implementation of Decision Maker with UTA decision function."""
"""Implementation of Decision Maker with UTA decision function."""

import matplotlib.pyplot as plt
import numpy as np


def get_random_uniform_normalized_vector(num_values, norm_value=1, decimals=3):
    """
    Functions that returns a random and normalized vector W
    W = [wi] such that wi >= 0, sum(wi) = norm_values, i = [1, num_values]
    """
    initial_vals = np.round(
        np.random.uniform(0, norm_value, num_values - 1), decimals=decimals
    ).astype("float32")
    initial_vals = np.sort(initial_vals)
    vect = [norm_value - initial_vals[-1]]
    for i in range(len(initial_vals) - 1):
        vect.append(initial_vals[-i - 1] - initial_vals[-i - 2])
    vect.append(initial_vals[0])
    return np.array(vect).astype("float32")


def piecewise_linear_value(criterion_value, marginal_coefficients, min_x=0, max_x=1):
    """
    Returns the value of the piecewise linear function defined by coeffs at x
    """
    # print(criterion_value, marginal_coefficients, min_x, max_x)
    assert criterion_value >= min_x and criterion_value <= max_x, (criterion_value, min_x, criterion_value, max_x)
    interval = (max_x - min_x) / (len(marginal_coefficients) - 1)
    for i in range(len(marginal_coefficients) - 1):
        if criterion_value >= min_x + i * interval and criterion_value <= min_x + (i + 1) * interval:
            return marginal_coefficients[i] + (marginal_coefficients[i + 1] - marginal_coefficients[i]) / interval * (
                criterion_value - min_x - i * interval
            )


def create_piecewise_linear_coefficients(n_pieces, min_criterion=0.0, max_criterion=1.0, n_decimals=5):
    coefficients = np.round(np.random.uniform(min_criterion, max_criterion, n_pieces - 1), decimals=n_decimals)
    coefficients = np.sort(coefficients)
    coefficients = np.concatenate([[min_criterion], coefficients, [max_criterion]]).astype("float32")
    return coefficients

"""    @np.vectorize
    def piecewise_linear_f(x):
        return piecewise_linear_value(
            x=x, coeffs=coefficients, min_x=min_x, max_x=max_x
        )

    return piecewise_linear_f, coefficients"""


class DecisionMaker:

    def __init__(self, n_criteria, n_pieces, n_decimals=5):
        self.n_criteria = n_criteria
        self.n_pieces = n_pieces
        self.n_decimals = n_decimals

        self.coefficients = self.build_random_decision_function()
        self.breakpoints_x = np.linspace(0, 1., self.n_pieces+1)
        self.total_n_answers = 0

    """def build_random_decision_function(self):
        slopes = []
        for i in range(self.n_criteria):
            crit_slopes = []
            for j in range(self.n_pieces):
                if i == 0 and j == 0:
                    crit_slopes.append(1.)
                else:
                    crit_slopes.append(np.random.uniform(0, 10))
            slopes.append(crit_slopes)
        return np.stack(slopes)"""
    def build_random_decision_function(self):
        coefficients = []
        for i in range(self.n_criteria):
            marginal_coefficients = create_piecewise_linear_coefficients(n_pieces=self.n_pieces,
            min_criterion=0., max_criterion=1., n_decimals=self.n_decimals)
            coefficients.append(marginal_coefficients)
        marginal_weights = get_random_uniform_normalized_vector(num_values=self.n_criteria, norm_value=1, decimals=2)
        while len(np.where(marginal_weights == 0)[0]) > 0:
            marginal_weights = get_random_uniform_normalized_vector(num_values=self.n_criteria, norm_value=1, decimals=2)
        return np.round(np.array(coefficients), self.n_decimals) * np.expand_dims(marginal_weights, axis=1)

    def load_from_parameters(self, coefficients, breakpoints_x=None):
        self.coefficients = coefficients
        # self.breakpoints_x = breakpoints_x


    def plot_decision_function(self, show=True):
        x = np.linspace(0, 1, self.n_pieces+1)

        plt.figure(figsize=(12, 4 * (self.n_criteria // 2 + self.n_criteria % 2)))
        for i in range(self.n_criteria):
            plt.subplot(2, self.n_criteria // 2 + self.n_criteria % 2, i+1)
            plt.plot(x, self.coefficients[i], label=f'Criterion {i+1}')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("Utility")
        if show:
            plt.show()

    def get_marginal_utility(self, criterion_index, criterion_value):
        return piecewise_linear_value(criterion_value, self.coefficients[criterion_index], min_x=self.breakpoints_x[0], max_x=self.breakpoints_x[-1])

    def get_total_utility(self, criteria_vector):
        total_utility = 0
        for criterion_index in range(self.n_criteria):
            total_utility += self.get_marginal_utility(criterion_index=criterion_index, criterion_value=criteria_vector[criterion_index])
        return total_utility

    def get_indifference_on_two_criteria(self, criterion_i, criterion_j, query_i, p_i, query_j):
        marginal_utility_difference_i = self.get_marginal_utility(criterion_i, p_i) - self.get_marginal_utility(criterion_i, query_i)
        marginal_utility_value_j = self.get_marginal_utility(criterion_j, query_j)

        self.total_n_answers += 1
        # print(marginal_utility_difference_i, marginal_utility_value_j)

        if marginal_utility_difference_i > 0:
            # Impossible to compensate the utility difference 
            if marginal_utility_difference_i > self.coefficients[criterion_j][-1] - marginal_utility_value_j:
                return None
            else:
                for break_point in range(self.n_pieces):
                    min_bp_value = self.coefficients[criterion_j][break_point]
                    max_bp_value = self.coefficients[criterion_j][break_point + 1]
                    if marginal_utility_value_j - marginal_utility_difference_i >= min_bp_value and marginal_utility_value_j - marginal_utility_difference_i < max_bp_value:
                        
                        dv = marginal_utility_value_j - min_bp_value - marginal_utility_difference_i
                        return self.breakpoints_x[break_point] + (dv / (max_bp_value - min_bp_value)) * (self.breakpoints_x[break_point+1] - self.breakpoints_x[break_point])
        else:
            if -marginal_utility_difference_i > (self.coefficients[criterion_j][-1] - marginal_utility_value_j):
                return None
            else:
                for break_point in range(self.n_pieces):
                    max_bp_value = self.coefficients[criterion_j][break_point+1]
                    min_bp_value = self.coefficients[criterion_j][break_point]
                    
                    if max_bp_value - marginal_utility_value_j > -marginal_utility_difference_i and min_bp_value - marginal_utility_value_j <= -marginal_utility_difference_i:
                        
                        dv = - marginal_utility_difference_i + marginal_utility_value_j - min_bp_value

                        return self.breakpoints_x[break_point] + dv / (max_bp_value - min_bp_value) * (self.breakpoints_x[break_point+1] - self.breakpoints_x[break_point])
        
    def get_total_n_answers(self):
        return self.total_n_answers

class HiddenDecisionMakers:

    def __init__(self, n_dms, n_criteria, n_pieces):
        self.dms = [DecisionMaker(n_criteria, n_pieces) for i in range(n_dms)]

    def query(self, criterion_i, criterion_j, query_i, p_i, query_j):
        answers = [dm.get_indifference_on_two_criteria(criterion_i=criterion_i,
            criterion_j=criterion_j,
            query_i=query_i, p_i=p_i, query_j=query_j) for dm in self.dms]

        return np.random.permutation(answers)

    def get_total_n_answers(self):
        return [dm.get_total_n_answers() for dm in self.dms]
