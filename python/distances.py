import gurobipy as gp
import numpy as np

class WorstUTA(object):
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
        self, X, Y, ref_coeffs, sample_weight=None, time_limit=None, verbose=0, majoring_value=10,
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

        self.abs_vals = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        self.abs_id = {
            (i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{i}_{j}")
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
                >= self.epsilon
            )
            for i in range(n_samples)
        }

        for j in range(self.n_pieces + 1):
            for i in range(n_features):
                self.solver.addConstr(self.abs_vals[(i, j)] >= self.marginal_coeffs[(i, j)] - ref_coeffs[(i, j)], name=f"a_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] >= - self.marginal_coeffs[(i, j)] + ref_coeffs[(i, j)], name=f"b_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  self.marginal_coeffs[(i, j)] - ref_coeffs[(i, j)] + majoring_value * self.abs_id[(i, j)], name=f"c_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  -self.marginal_coeffs[(i, j)] + ref_coeffs[(i, j)] + majoring_value * (1 - self.abs_id[(i, j)]), name=f"d_{i}_{j}")

        ### Obj

        if sample_weight is not None:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] * sample_weight[i] for i in range(n_samples)),
                gp.GRB.MINIMIZE,
            )
        else:
            self.solver.setObjective(
                gp.quicksum(self.abs_vals[(i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), gp.GRB.MAXIMIZE
            )
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status

        if self.status == 2:
            self.coeffs = [
                [self.marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]

            for k, v in estimate_x.items():
                estimate_x[k] = v.x
            for k, v in estimate_y.items():
                estimate_y[k] = v.x

            return estimate_x, estimate_y
        else:
            return "error"

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


    def fit_from_indifferences(
        self, X, Y, ref_coeffs, sample_weight=None, time_limit=None, verbose=0, majoring_value=10,
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

        self.abs_vals = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        self.abs_id = {
            (i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{i}_{j}")
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
        # Indifferences
        pref_A = {
            (i): self.solver.addConstr(
                gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                - gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                <= self.epsilon
            )
            for i in range(n_samples)
        }
        pref_B = {
            (i): self.solver.addConstr(
                - gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                + gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                <= self.epsilon
            )
            for i in range(n_samples)
        }

        for j in range(self.n_pieces + 1):
            for i in range(n_features):
                self.solver.addConstr(self.abs_vals[(i, j)] >= self.marginal_coeffs[(i, j)] - ref_coeffs[(i, j)], name=f"a_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] >= - self.marginal_coeffs[(i, j)] + ref_coeffs[(i, j)], name=f"b_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  self.marginal_coeffs[(i, j)] - ref_coeffs[(i, j)] + majoring_value * self.abs_id[(i, j)], name=f"c_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  -self.marginal_coeffs[(i, j)] + ref_coeffs[(i, j)] + majoring_value * (1 - self.abs_id[(i, j)]), name=f"d_{i}_{j}")

        ### Obj

        if sample_weight is not None:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] * sample_weight[i] for i in range(n_samples)),
                gp.GRB.MINIMIZE,
            )
        else:
            self.solver.setObjective(
                gp.quicksum(self.abs_vals[(i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), gp.GRB.MAXIMIZE
            )
        self.solver.update()

        # -- Résolution --
        print("Optimize")
        self.solver.optimize()
        self.status = self.solver.Status
        print(self.status)

        if self.status == 2:
            self.coeffs = [
                [self.marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]

            for k, v in estimate_x.items():
                estimate_x[k] = v.x
            for k, v in estimate_y.items():
                estimate_y[k] = v.x

            return estimate_x, estimate_y
        else:
            return "error"


class UTASpaceDiameter(object):
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
        self, X, Y, sample_weight=None, time_limit=None, verbose=0, majoring_value=10,
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
        lower_bound_marginal_coeffs = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        upper_bound_marginal_coeffs = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }

        self.abs_vals = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        self.abs_id = {
            (i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }

        self.lower_bound_marginal_coeffs = lower_bound_marginal_coeffs
        self.upper_bound_marginal_coeffs = upper_bound_marginal_coeffs

        lower_estimate_x = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }
        lower_estimate_y = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }

        upper_estimate_x = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }
        upper_estimate_y = {
            (i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
        }


        if verbose > 1:
            print("2/ Constraints Definition")
        # [MI - 2]
        for name, marginal_coeffs, estimate_x, estimate_y in zip(
            ["low", "high"],
            [self.lower_bound_marginal_coeffs, self.upper_bound_marginal_coeffs],
            [lower_estimate_x, upper_estimate_x],
            [lower_estimate_y, upper_estimate_y]
        ):
            zero_bound = {
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
                (name, i): self.solver.addConstr(
                    gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                    - gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                    >= self.epsilon
                )
                for i in range(n_samples)
            }

        for j in range(self.n_pieces + 1):
            for i in range(n_features):
                self.solver.addConstr(self.abs_vals[(i, j)] >= self.lower_bound_marginal_coeffs[(i, j)] - self.upper_bound_marginal_coeffs[(i, j)], name=f"a_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] >= - self.lower_bound_marginal_coeffs[(i, j)] + self.upper_bound_marginal_coeffs[(i, j)], name=f"b_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  self.lower_bound_marginal_coeffs[(i, j)] - self.upper_bound_marginal_coeffs[(i, j)] + majoring_value * self.abs_id[(i, j)], name=f"c_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  -self.lower_bound_marginal_coeffs[(i, j)] + self.upper_bound_marginal_coeffs[(i, j)] + majoring_value * (1 - self.abs_id[(i, j)]), name=f"d_{i}_{j}")

        ### Obj

        if sample_weight is not None:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] * sample_weight[i] for i in range(n_samples)),
                gp.GRB.MINIMIZE,
            )
        else:
            self.solver.setObjective(
                gp.quicksum(self.abs_vals[(i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), gp.GRB.MAXIMIZE
            )
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status

        if self.status == 2:
            self.lower_bound_coeffs = [
                [self.lower_bound_marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]
            self.higher_bound_coeffs = [
                [self.upper_bound_marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]

            """for k, v in estimate_x.items():
                estimate_x[k] = v.x
            for k, v in estimate_y.items():
                estimate_y[k] = v.x

            return estimate_x, estimate_y"""
            return
        else:
            return "error"

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


    def fit_from_indifferences(
        self, X, Y, ref_coeffs, sample_weight=None, time_limit=None, verbose=0, majoring_value=10,
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

        self.abs_vals = {
            (i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
        }
        self.abs_id = {
            (i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{i}_{j}")
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
        # Indifferences
        pref_A = {
            (i): self.solver.addConstr(
                gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                - gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                <= self.epsilon
            )
            for i in range(n_samples)
        }
        pref_B = {
            (i): self.solver.addConstr(
                - gp.quicksum([estimate_x[(i, j)] for j in range(n_features)])
                + gp.quicksum([estimate_y[(i, j)] for j in range(n_features)])
                <= self.epsilon
            )
            for i in range(n_samples)
        }

        for j in range(self.n_pieces + 1):
            for i in range(n_features):
                self.solver.addConstr(self.abs_vals[(i, j)] >= self.marginal_coeffs[(i, j)] - ref_coeffs[(i, j)], name=f"a_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] >= - self.marginal_coeffs[(i, j)] + ref_coeffs[(i, j)], name=f"b_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  self.marginal_coeffs[(i, j)] - ref_coeffs[(i, j)] + majoring_value * self.abs_id[(i, j)], name=f"c_{i}_{j}")
                self.solver.addConstr(self.abs_vals[(i, j)] <=  -self.marginal_coeffs[(i, j)] + ref_coeffs[(i, j)] + majoring_value * (1 - self.abs_id[(i, j)]), name=f"d_{i}_{j}")

        ### Obj

        if sample_weight is not None:
            self.solver.setObjective(
                gp.quicksum(sigma_err[i] * sample_weight[i] for i in range(n_samples)),
                gp.GRB.MINIMIZE,
            )
        else:
            self.solver.setObjective(
                gp.quicksum(self.abs_vals[(i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), gp.GRB.MAXIMIZE
            )
        self.solver.update()

        # -- Résolution --
        print("Optimize")
        self.solver.optimize()
        self.status = self.solver.Status
        print(self.status)

        if self.status == 2:
            self.coeffs = [
                [self.marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]

            for k, v in estimate_x.items():
                estimate_x[k] = v.x
            for k, v in estimate_y.items():
                estimate_y[k] = v.x

            return estimate_x, estimate_y
        else:
            return "error"


class TwoUTASpaceDiameter(object):
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
        self, X, Y, sample_weight=None, time_limit=None, verbose=0, majoring_value=10,
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
        self.marginal_coeffs = {    
            (name, i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["d1", "d2", "s1", "s2"]
        }

        self.abs_vals = {
            (name, i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["d1_s1", "d1_s2", "d2_s1", "d2_s2"]
        }
        self.abs_id = {
            (name, i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["d1_s1", "d1_s2", "d2_s1", "d2_s2"]
        }

        self.estimate_x = {
            (name, i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{name}_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
            for name in ["d1", "d2", "s1", "s2"]
        }
        self.estimate_y = {
            (name, i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{name}_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
            for name in ["d1", "d2", "s1", "s2"]
        }

        self.z_s = {
            k: self.solver.addVar(
                vtype=gp.GRB.BINARY, name=f"z_s_{k}"
            )
            for k in range(n_samples)
        }
        self.z_d = {
            k: self.solver.addVar(
                vtype=gp.GRB.BINARY, name=f"z_d_{k}"
            )
            for k in range(n_samples)
        }

        if verbose > 1:
            print("2/ Constraints Definition")
        # [MI - 2]
        for name in ["d1", "d2", "s1", "s2"]:
            zero_bound = {
                i: self.solver.addConstr(
                    self.marginal_coeffs[name, i, 0] == 0, name="lower_bound_normalization"
                )
                for i in range(n_features)
            }
            sum_to_one = {
                self.solver.addConstr(
                    (
                        gp.quicksum(
                            self.marginal_coeffs[name, i, self.n_pieces] for i in range(n_features)
                        )
                    )
                    == 1,
                    name="sum_to_one",
                )
            }
            monotonicity = {
                (i): {
                    k: self.solver.addConstr(
                        self.marginal_coeffs[name, i, k + 1] >= self.marginal_coeffs[name, i, k],
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
                                self.estimate_x[name, i, j]
                                == self.marginal_coeffs[name, j, k]
                                + (
                                    (X[i][j] - self.inflexions[j][k])
                                    / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                                )
                                * (self.marginal_coeffs[name, j, k + 1] - self.marginal_coeffs[name, j, k]),
                                name=f"estimate_x_{i}_{j}",
                            )
                        if self.inflexions[j][k] <= Y[i][j] <= self.inflexions[j][k + 1]:
                            self.solver.addConstr(
                                self.estimate_y[name, i, j]
                                == self.marginal_coeffs[name, j, k]
                                + (
                                    (Y[i][j] - self.inflexions[j][k])
                                    / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                                )
                                * (self.marginal_coeffs[name, j, k + 1] - self.marginal_coeffs[name, j, k]),
                                name=f"estimate_y_{i}_{j}",
                            )
            
        pref_d1 = {
            ("d1", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("d1", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("d1", i, j)] for j in range(n_features)])
                + majoring_value * self.z_d[i]
                >= self.epsilon
            )
            for i in range(n_samples)
        }
        pref_d2 = {
            ("d2", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("d2", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("d2", i, j)] for j in range(n_features)])
                + majoring_value * (1 - self.z_d[i])
                >= self.epsilon
            )
            for i in range(n_samples)
        }

        pref_s1 = {
            ("s1", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("s1", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("s1", i, j)] for j in range(n_features)])
                + majoring_value * self.z_s[i]
                >= self.epsilon
            )
            for i in range(n_samples)
        }
        pref_s2 = {
            ("s2", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("s2", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("s2", i, j)] for j in range(n_features)])
                + majoring_value * (1 - self.z_s[i])
                >= self.epsilon
            )
            for i in range(n_samples)
        }

        for name in [("d1", "s1"), ("d1", "s2"), ("d2", "s1"), ("d2", "s2")]:
            for j in range(self.n_pieces + 1):
                for i in range(n_features):
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] >= self.marginal_coeffs[(name[0], i, j)] - self.marginal_coeffs[(name[1], i, j)], name=f"a_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] >= - self.marginal_coeffs[(name[0], i, j)] + self.marginal_coeffs[(name[1], i, j)], name=f"b_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] <=  self.marginal_coeffs[(name[0], i, j)] - self.marginal_coeffs[(name[1], i, j)] + majoring_value * self.abs_id[(f"{name[0]}_{name[1]}", i, j)], name=f"c_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] <=  - self.marginal_coeffs[(name[0], i, j)] + self.marginal_coeffs[(name[1], i, j)] + majoring_value * (1 - self.abs_id[(f"{name[0]}_{name[1]}", i, j)]), name=f"d_{i}_{j}")

        ### Obj

        distance = self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"D"
            )
        self.solver.addConstr(distance <= gp.quicksum(self.abs_vals[("d1_s1", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)) + gp.quicksum(self.abs_vals[("d2_s2", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), name=f"xyz")
        self.solver.addConstr(distance <= gp.quicksum(self.abs_vals[("d2_s1", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)) + gp.quicksum(self.abs_vals[("d1_s2", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), name=f"xyz")

        self.solver.setObjective(
            distance, gp.GRB.MAXIMIZE
        )
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status

        """if self.status == 2:
            self.lower_bound_coeffs = [
                [self.lower_bound_marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]
            self.higher_bound_coeffs = [
                [self.upper_bound_marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]
            '''
            for k, v in estimate_x.items():
                estimate_x[k] = v.x
            for k, v in estimate_y.items():
                estimate_y[k] = v.x

            return estimate_x, estimate_y'''
            return
        else:
            return "error"

        """

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


    def fit_from_indifferences(
        self, X, Y, sample_weight=None, time_limit=None, verbose=0, majoring_value=10,
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
        self.marginal_coeffs = {    
            (name, i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["d1", "d2", "s1", "s2"]
        }

        self.abs_vals = {
            (name, i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["d1_s1", "d1_s2", "d2_s1", "d2_s2"]
        }
        self.abs_id = {
            (name, i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["d1_s1", "d1_s2", "d2_s1", "d2_s2"]
        }

        self.estimate_x = {
            (name, i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{name}_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
            for name in ["d1", "d2", "s1", "s2"]
        }
        self.estimate_y = {
            (name, i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{name}_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
            for name in ["d1", "d2", "s1", "s2"]
        }

        self.z_s = {
            k: self.solver.addVar(
                vtype=gp.GRB.BINARY, name=f"z_s_{k}"
            )
            for k in range(n_samples)
        }
        self.z_d = {
            k: self.solver.addVar(
                vtype=gp.GRB.BINARY, name=f"z_d_{k}"
            )
            for k in range(n_samples)
        }

        if verbose > 1:
            print("2/ Constraints Definition")
        # [MI - 2]
        for name in ["d1", "d2", "s1", "s2"]:
            zero_bound = {
                i: self.solver.addConstr(
                    self.marginal_coeffs[name, i, 0] == 0, name="lower_bound_normalization"
                )
                for i in range(n_features)
            }
            sum_to_one = {
                self.solver.addConstr(
                    (
                        gp.quicksum(
                            self.marginal_coeffs[name, i, self.n_pieces] for i in range(n_features)
                        )
                    )
                    == 1,
                    name="sum_to_one",
                )
            }
            monotonicity = {
                (i): {
                    k: self.solver.addConstr(
                        self.marginal_coeffs[name, i, k + 1] >= self.marginal_coeffs[name, i, k],
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
                                self.estimate_x[name, i, j]
                                == self.marginal_coeffs[name, j, k]
                                + (
                                    (X[i][j] - self.inflexions[j][k])
                                    / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                                )
                                * (self.marginal_coeffs[name, j, k + 1] - self.marginal_coeffs[name, j, k]),
                                name=f"estimate_x_{i}_{j}",
                            )
                        if self.inflexions[j][k] <= Y[i][j] <= self.inflexions[j][k + 1]:
                            self.solver.addConstr(
                                self.estimate_y[name, i, j]
                                == self.marginal_coeffs[name, j, k]
                                + (
                                    (Y[i][j] - self.inflexions[j][k])
                                    / (self.inflexions[j][k + 1] - self.inflexions[j][k])
                                )
                                * (self.marginal_coeffs[name, j, k + 1] - self.marginal_coeffs[name, j, k]),
                                name=f"estimate_y_{i}_{j}",
                            )
            
        pref_d1 = {
            ("d1", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("d1", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("d1", i, j)] for j in range(n_features)])
                - majoring_value * self.z_d[i]
                <= self.epsilon
            )
            for i in range(n_samples)
        }
        pref_d2 = {
            ("d2", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("d2", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("d2", i, j)] for j in range(n_features)])
                - majoring_value * (1 - self.z_d[i])
                <= self.epsilon
            )
            for i in range(n_samples)
        }

        pref_s1 = {
            ("s1", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("s1", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("s1", i, j)] for j in range(n_features)])
                - majoring_value * self.z_s[i]
                <= self.epsilon
            )
            for i in range(n_samples)
        }
        pref_s2 = {
            ("s2", i): self.solver.addConstr(
                gp.quicksum([self.estimate_x[("s2", i, j)] for j in range(n_features)])
                - gp.quicksum([self.estimate_y[("s2", i, j)] for j in range(n_features)])
                - majoring_value * (1 - self.z_s[i])
                <= self.epsilon
            )
            for i in range(n_samples)
        }

        for name in [("d1", "s1"), ("d1", "s2"), ("d2", "s1"), ("d2", "s2")]:
            for j in range(self.n_pieces + 1):
                for i in range(n_features):
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] >= self.marginal_coeffs[(name[0], i, j)] - self.marginal_coeffs[(name[1], i, j)], name=f"a_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] >= - self.marginal_coeffs[(name[0], i, j)] + self.marginal_coeffs[(name[1], i, j)], name=f"b_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] <=  self.marginal_coeffs[(name[0], i, j)] - self.marginal_coeffs[(name[1], i, j)] + majoring_value * self.abs_id[(f"{name[0]}_{name[1]}", i, j)], name=f"c_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] <=  - self.marginal_coeffs[(name[0], i, j)] + self.marginal_coeffs[(name[1], i, j)] + majoring_value * (1 - self.abs_id[(f"{name[0]}_{name[1]}", i, j)]), name=f"d_{i}_{j}")

        ### Obj

        distance = self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"D"
            )
        self.solver.addConstr(distance <= gp.quicksum(self.abs_vals[("d1_s1", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)) + gp.quicksum(self.abs_vals[("d2_s2", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), name=f"xyz")
        self.solver.addConstr(distance <= gp.quicksum(self.abs_vals[("d2_s1", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)) + gp.quicksum(self.abs_vals[("d1_s2", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), name=f"xyz")

        self.solver.setObjective(
            distance, gp.GRB.MAXIMIZE
        )
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status

        """if self.status == 2:
            self.lower_bound_coeffs = [
                [self.lower_bound_marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]
            self.higher_bound_coeffs = [
                [self.upper_bound_marginal_coeffs[(i, j)].x for j in range(self.n_pieces + 1)]
                for i in range(n_features)
            ]
            '''
            for k, v in estimate_x.items():
                estimate_x[k] = v.x
            for k, v in estimate_y.items():
                estimate_y[k] = v.x

            return estimate_x, estimate_y'''
            return
        else:
            return "error"

        """