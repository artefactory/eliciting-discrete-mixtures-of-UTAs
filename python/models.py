import os

import gurobipy as gp
import numpy as np


class UTA(object):
    """Gurobi based implementation of UTA."""

    def __init__(self, n_pieces, epsilon=1e-4):
        """Initialize Model.

        Parameters:
        -----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        """
        self.seed = 123
        self.n_pieces = n_pieces
        self.epsilon = epsilon
        self.solver = self.instantiate()

    def _determine_inflexions(self, X, Y):
        """Determine inflexions of the utility functions for each feature.

        Parameters:
        -----------
        X: np.ndarray
            preferred elements
        Y: np.ndarray
            non preferred elements

        Return:
        --------
            list of lists of floats: inflexions for each feature
        """
        minimal_values = np.min(np.min([X, Y], axis=0), axis=0)
        maximal_values = np.max(np.max([X, Y], axis=0), axis=0)

        inflexions = []
        for criterion_nb in range(X.shape[1]):
            if maximal_values[criterion_nb] == minimal_values[criterion_nb]:
                dval = 1e-4 * self.n_pieces / 2
                maximal_values[criterion_nb] += dval
                minimal_values[criterion_nb] -= dval
            crit_inflexions = []
            for i in range(self.n_pieces):
                crit_inflexions.append(
                    minimal_values[criterion_nb]
                    + i
                    * (
                        (maximal_values[criterion_nb] - minimal_values[criterion_nb])
                        / self.n_pieces
                    )
                )
            crit_inflexions.append(maximal_values[criterion_nb])
            inflexions.append(crit_inflexions)
        return minimal_values, maximal_values, inflexions

    def _get_marginal_utility(self, val, coeffs, criterion_nb):
        r"""Returns a marginal (1D) utility in the form of a piece-wise linear function.

        Paramters:
        ----------
        val: float
            value of the feature for which we want to compute the marginal utility. min_ <= val <= max_
        coeffs: list of floats
            coefficients of the piece-wise linear function
        min_: float
            minimum value of the feature
        max_: float
            maximum value of the feature

        Returns:
        --------
            float: piece-wise linear marginal utility: \sum_i coeffs[i] * max(0, min(val - inflexions_x[i], inflexions_x[i + 1] - inflexions_x[i]))
        """
        inflexions_x = self.inflexions[criterion_nb]

        for i in range(self.n_pieces):
            if inflexions_x[i] <= val <= inflexions_x[i + 1]:
                return coeffs[i] + (
                    (val - inflexions_x[i]) / (inflexions_x[i + 1] - inflexions_x[i])
                ) * (coeffs[i + 1] - coeffs[i])
        # print(f"val {val}is not in the range of the utility function:{inflexions_x}")
        if val < inflexions_x[0]:
            return 0
        elif val > inflexions_x[-1]:
            return coeffs[-1]
        else:
            raise ValueError("Something went wrong")

    def instantiate(self):
        """Instantiate the solver"""
        solver = gp.Model("UTA")
        return solver

    def fit(
        self, X, Y, inflexions=None, sample_weight=None, time_limit=None, verbose=0
    ):
        """Estimation of the parameters"""
        n_samples = X.shape[0]
        n_features = Y.shape[1]
        
        self.min, self.max, self.inflexions = self._determine_inflexions(X, Y)

        if verbose == 0:
            self.solver.params.outputflag = 0  # mode muet

        if time_limit is not None:
            self.solver.setParam("TimeLimit", time_limit)

        if verbose > 1:
            print("1/ Variables Definition")
        marginal_coeffs = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        self.marginal_coeffs = marginal_coeffs
        estimate_x = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }
        estimate_y = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }

        sigma_err = {
            i: self.solver.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"noise-pos_{i}")
            for i in range(n_samples)
        }

        if verbose > 1:
            print("2/ Constraints Definition")
        # [MI - 2]
        lower_bound = {
            i: self.solver.addConstr(
                marginal_coeffs[i, 0] == 0, name="lower_bound_normalization"
            )
            for i in range(n_features)
        }
        sum_to_one = {
            self.solver.addConstr(
                (
                    gp.quicksum(
                        marginal_coeffs[i, self.n_pieces] for i in range(n_features)
                    )
                )
                == 1,
                name="sum_to_one",
            )
        }
        monotonicity = {
            (i): {
                k: self.solver.addConstr(
                    marginal_coeffs[i, k + 1] >= marginal_coeffs[i, k],
                    name="monotonicity",
                )
                for k in range(self.n_pieces)
            }
            for i in range(n_features)
        }
        # [MI - 4]
        # [MI - 1']
        # [MI - 8']
        if verbose > 1:
            print("3/ Utility Constraints")
        # Contraintes de préférences
        for i in range(n_samples):
            for j in range(n_features):
                for k in range(self.n_pieces):
                    if self.inflexions[j][k] <= X[i][j] <= self.inflexions[j][k + 1]:
                        self.solver.addConstr(
                            estimate_x[i, j]
                            == marginal_coeffs[j, k]
                            + (
                                (X[i][j] - self.inflexions[j][k])
                                / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                            )
                            * (marginal_coeffs[j, k + 1] - marginal_coeffs[j, k]),
                            name=f"estimate_x_{i}_{j}",
                        )
                    if self.inflexions[j][k] <= Y[i][j] <= self.inflexions[j][k + 1]:
                        self.solver.addConstr(
                            estimate_y[i, j]
                            == marginal_coeffs[j, k]
                            + (
                                (Y[i][j] - self.inflexions[j][k])
                                / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                            )
                            * (marginal_coeffs[j, k + 1] - marginal_coeffs[j, k]),
                            name=f"estimate_y_{i}_{j}",
                        )
        pref = {
            (i): self.solver.addConstr(
                gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                - gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                + sigma_err[i]
                >= self.epsilon
            )
            for i in range(n_samples)
        }
        if sample_weight is not None:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] * sample_weight[i] for i in range(n_samples)),
                gp.GRB.MINIMIZE,
            )
        else:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] for i in range(n_samples)), gp.GRB.MINIMIZE
            )
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status

        self.coeffs = [
            [self.marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
            for i in range(n_features)
        ]
        for k, v in sigma_err.items():
            sigma_err[k] = v.x

        for k, v in estimate_x.items():
            estimate_x[k] = v.x
        for k, v in estimate_y.items():
            estimate_y[k] = v.x

        return sigma_err, estimate_x, estimate_y

    def fit_from_indifferences(
        self, X, Y, inflexions=None, sample_weight=None, time_limit=None, verbose=0
    ):
        """Estimation of the parameters"""
        n_samples = X.shape[0]
        n_features = Y.shape[1]

        if inflexions is None:
            self.min, self.max, self.inflexions = self._determine_inflexions(X, Y)
        else:
            inflexions = np.array(inflexions)

            assert inflexions.shape[0] == n_features
            assert inflexions.shape[1] == self.n_pieces + 1

            self.min = np.min(inflexions, axis=0)
            self.max = np.max(inflexions, axis=0)
            self.inflexions = inflexions

        if verbose == 0:
            self.solver.params.outputflag = 0  # mode muet

        if time_limit is not None:
            self.solver.setParam("TimeLimit", time_limit)

        if verbose > 1:
            print("1/ Variables Definition")
        marginal_coeffs = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        self.marginal_coeffs = marginal_coeffs
        estimate_x = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }
        estimate_y = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }

        sigma_err = {
            i: self.solver.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, name=f"noise-pos_{i}")
            for i in range(n_samples)
        }

        if verbose > 1:
            print("2/ Constraints Definition")
        # [MI - 2]
        lower_bound = {
            i: self.solver.addConstr(
                marginal_coeffs[i, 0] == 0, name="lower_bound_normalization"
            )
            for i in range(n_features)
        }
        sum_to_one = {
            self.solver.addConstr(
                (
                    gp.quicksum(
                        marginal_coeffs[i, self.n_pieces] for i in range(n_features)
                    )
                )
                == 1,
                name="sum_to_one",
            )
        }
        monotonicity = {
            (i): {
                k: self.solver.addConstr(
                    marginal_coeffs[i, k + 1] >= marginal_coeffs[i, k],
                    name="monotonicity",
                )
                for k in range(self.n_pieces)
            }
            for i in range(n_features)
        }
        # [MI - 4]
        # [MI - 1']
        # [MI - 8']
        if verbose > 1:
            print("3/ Utility Constraints")
        # Contraintes de préférences
        for i in range(n_samples):
            for j in range(n_features):
                for k in range(self.n_pieces):
                    if self.inflexions[j][k] <= X[i][j] <= self.inflexions[j][k + 1]:
                        self.solver.addConstr(
                            estimate_x[i, j]
                            == marginal_coeffs[j, k]
                            + (
                                (X[i][j] - self.inflexions[j][k])
                                / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                            )
                            * (marginal_coeffs[j, k + 1] - marginal_coeffs[j, k]),
                            name=f"estimate_x_{i}_{j}",
                        )
                    if self.inflexions[j][k] <= Y[i][j] <= self.inflexions[j][k + 1]:
                        self.solver.addConstr(
                            estimate_y[i, j]
                            == marginal_coeffs[j, k]
                            + (
                                (Y[i][j] - self.inflexions[j][k])
                                / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                            )
                            * (marginal_coeffs[j, k + 1] - marginal_coeffs[j, k]),
                            name=f"estimate_y_{i}_{j}",
                        )
        self.indiff_1 = {
            (i): self.solver.addConstr(
                gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                - gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                - sigma_err[i]
                <= self.epsilon
            )
            for i in range(n_samples)
        }
        self.indiff_2 = {
            (i): self.solver.addConstr(
                gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                - gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                - sigma_err[i]
                <= self.epsilon
            )
            for i in range(n_samples)
        }
        if sample_weight is not None:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] * sample_weight[i] for i in range(n_samples)),
                gp.GRB.MINIMIZE,
            )
        else:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] for i in range(n_samples)), gp.GRB.MINIMIZE
            )
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status

        self.coeffs = [
            [self.marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
            for i in range(n_features)
        ]
        for k, v in sigma_err.items():
            sigma_err[k] = v.x

        for k, v in estimate_x.items():
            estimate_x[k] = v.x
        for k, v in estimate_y.items():
            estimate_y[k] = v.x

        return sigma_err, estimate_x, estimate_y

    def _get_total_utility(self, X, coeffs):
        utilities = []
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        for x in X:
            utility = 0
            for i in range(len(coeffs)):
                utility += self._get_marginal_utility(x[i], coeffs[i], i)
            utilities.append(utility)
        return np.squeeze(utilities)

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X."""
        utilities = self._get_total_utility(X, self.coeffs)

        return utilities
