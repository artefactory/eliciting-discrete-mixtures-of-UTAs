import json
import os

import gurobipy as gp
import numpy as np


class TwoUTASpaceRadius(object):
    """Gurobi based implementation of the radius."""

    def __init__(self, n_pieces, epsilon=1e-4, lipschitz_coeff=0., focus_on_solution=False):
        """Initialize Model.

        Parameters:
        -----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        """
        self.seed = 123
        self.n_pieces = n_pieces
        self.epsilon = epsilon
        self.lipschitz_coeff = lipschitz_coeff
        self.focus_on_solution = focus_on_solution
        self.solver = self.instantiate()

        self.optim_params = {}

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

    def fit_generic(
        self,
        X,
        Y,
        space_center,
        relation_type="preference",
        unclustering=None,
        inflexions=None,
        sample_weight=None,
        time_limit=None,
        n_threads=None,
        verbose=0,
        majoring_value=10,
        warm_coefficients=None,
        warm_zs=None,
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

        if unclustering is not None:
            values, counts = np.unique(unclustering, return_counts=True)
            for count in counts:
                assert count == 2
            n_couples = int(n_samples // 2)
            print("N couples:", n_couples)
        else:
            n_couples = len(X)
            unclustering = np.arange(0, len(X))

        if verbose == 0:
            self.solver.params.outputflag = 0  # mode muet

        if time_limit is not None:
            self.solver.setParam("TimeLimit", time_limit)
        if n_threads is not None:
            self.solver.setParam("Threads", n_threads)

        if verbose > 1:
            print("1/ Variables Definition")
        self.marginal_coeffs = {    
            (name, i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"s_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["r1", "r2"]
        }

        if warm_coefficients is not None:
            for key, val in self.marginal_coeffs.items():
                setattr(val, "Start", warm_coefficients[key])        

        self.abs_vals = {
            (name, i, j): self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name=f"absval_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["r1_c1", "r1_c2", "r2_c1", "r2_c2"]
        }
        self.abs_id = {
            (name, i, j): self.solver.addVar(vtype=gp.GRB.BINARY, name=f"absid_{name}_{i}_{j}")
            for j in range(self.n_pieces + 1)
            for i in range(n_features)
            for name in ["r1_c1", "r1_c2", "r2_c1", "r2_c2"]
        }

        self.estimate_x = {
            (name, i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_x_{name}_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
            for name in ["r1", "r2"]
        }
        self.estimate_y = {
            (name, i, j): self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"estimate_y_{name}_{i}_{j}"
            )
            for j in range(n_features)
            for i in range(n_samples)
            for name in ["r1", "r2"]
        }

        self.z_r = {
            k: self.solver.addVar(
                vtype=gp.GRB.BINARY, name=f"z_r_{k}"
            )
            for k in range(n_couples)
        }
        if warm_zs is not None:
            if len(X) == n_couples:
                for i, val in enumerate(self.z_r.values()):
                    if i % 2 == 0:
                        setattr(val, "Start", 0)
                    else:
                        setattr(val, "Start", 1)
            else:
                for val in self.z_r.values():
                    setattr(val, "Start", 0)
        if verbose > 1:
            print("2/ Constraints Definition")
        # [MI - 2]
        self.zero_bound = {}
        self.sum_to_one = {}
        self.monotonicity = {}
        for name in ["r1", "r2",]:
            for i in range(n_features):
                self.zero_bound[(name, i)] = self.solver.addConstr(
                        self.marginal_coeffs[name, i, 0] == 0, name="lower_bound_normalization"
                    )

            self.sum_to_one[name] = self.solver.addConstr(
                    (
                        gp.quicksum(
                            self.marginal_coeffs[name, i, self.n_pieces] for i in range(n_features)
                        )
                    )
                    == 1,
                    name="sum_to_one",
                )
            for i in range(n_features):
                self.monotonicity[(name, i)] = {
                        k: self.solver.addConstr(
                            self.marginal_coeffs[name, i, k + 1] >= self.marginal_coeffs[name, i, k] + self.lipschitz_coeff,
                            name="monotonicity",
                        )
                        for k in range(self.n_pieces)
                        }

            # [MI - 4]
            # [MI - 1']
            # [MI - 8']
            if verbose > 1:
                print("3/ Utility Constraints")
            # Contraintes de préferences
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
            
        self.pref_r1 = {}
        self.pref_r2 = {}
        for i in range(n_samples):
            if relation_type == "preference":
                if i % 2 == 0:
                    self.pref_r1["r1", i] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r1", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r1", i, j)] for j in range(n_features)])
                            + majoring_value * self.z_r[unclustering[i]]
                            >= self.epsilon, name=f"r1_pref_{i}"
                        )
                    
                    self.pref_r2["r2", i] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r2", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r2", i, j)] for j in range(n_features)])
                            + majoring_value * (1 - self.z_r[unclustering[i]])
                            >= self.epsilon, name=f"r2_pref_{i}"
                        )

                else:
                    self.pref_r1["r1", i] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r1", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r1", i, j)] for j in range(n_features)])
                            + majoring_value * (1 - self.z_r[unclustering[i]])
                            >= self.epsilon, name=f"r1_pref_{i}"
                        )

                    self.pref_r2["r2", i] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r2", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r2", i, j)] for j in range(n_features)])
                            + majoring_value * self.z_r[unclustering[i]]
                            >= self.epsilon, name=f"r2_pref_{i}"
                        )

            elif relation_type == "indifference":
                if i % 2 == 0:
                    self.pref_r1["r1", i, "xy"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r1", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r1", i, j)] for j in range(n_features)])
                            - majoring_value * self.z_r[unclustering[i]]
                            <= self.epsilon, name=f"r1_pref_{i}"
                        )
                    self.pref_r1["r1", i, "yx"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_y[("r1", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_x[("r1", i, j)] for j in range(n_features)])
                            - majoring_value * self.z_r[unclustering[i]]
                            <= self.epsilon, name=f"r1_pref_{i}"
                        )
                    
                    self.pref_r2["r2", i, "xy"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r2", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r2", i, j)] for j in range(n_features)])
                            - majoring_value * (1 - self.z_r[unclustering[i]])
                            <= self.epsilon, name=f"r2_pref_{i}"
                        )
                    self.pref_r2["r2", i, "yx"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_y[("r2", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_x[("r2", i, j)] for j in range(n_features)])
                            - majoring_value * (1 - self.z_r[unclustering[i]])
                            <= self.epsilon, name=f"r2_pref_{i}"
                        )

                else:
                    self.pref_r1["r1", i, "xy"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r1", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r1", i, j)] for j in range(n_features)])
                            - majoring_value * (1 - self.z_r[unclustering[i]])
                            <= self.epsilon, name=f"r1_pref_{i}"
                        )
                    self.pref_r1["r1", i, "yx"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_y[("r1", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_x[("r1", i, j)] for j in range(n_features)])
                            - majoring_value * (1 - self.z_r[unclustering[i]])
                            <= self.epsilon, name=f"r1_pref_{i}"
                        )

                    self.pref_r2["r2", i, "xy"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_x[("r2", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_y[("r2", i, j)] for j in range(n_features)])
                            - majoring_value * self.z_r[unclustering[i]]
                            <= self.epsilon, name=f"r2_pref_{i}"
                        )
                    self.pref_r2["r2", i, "yx"] = self.solver.addConstr(
                            gp.quicksum([self.estimate_y[("r2", i, j)] for j in range(n_features)])
                            - gp.quicksum([self.estimate_x[("r2", i, j)] for j in range(n_features)])
                            - majoring_value * self.z_r[unclustering[i]]
                            <= self.epsilon, name=f"r2_pref_{i}"
                        )

            else:
                raise ValueError("Unknown relation_type")

        for name in [("r1", "c1"), ("r1", "c2"), ("r2", "c1"), ("r2", "c2")]:
            for j in range(self.n_pieces + 1):
                for i in range(n_features):
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] >= self.marginal_coeffs[(name[0], i, j)] - space_center[(name[1], i, j)], name=f"a_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] >= - self.marginal_coeffs[(name[0], i, j)] + space_center[(name[1], i, j)], name=f"b_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] <=  self.marginal_coeffs[(name[0], i, j)] - space_center[(name[1], i, j)] + majoring_value * self.abs_id[(f"{name[0]}_{name[1]}", i, j)], name=f"c_{i}_{j}")
                    self.solver.addConstr(self.abs_vals[(f"{name[0]}_{name[1]}", i, j)] <=  - self.marginal_coeffs[(name[0], i, j)] + space_center[(name[1], i, j)] + majoring_value * (1 - self.abs_id[(f"{name[0]}_{name[1]}", i, j)]), name=f"d_{i}_{j}")

        ### Obj

        distance = self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"D"
            )
        self.solver.addConstr(distance <= gp.quicksum(self.abs_vals[("r1_c1", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)) + gp.quicksum(self.abs_vals[("r2_c2", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), name=f"xyz")
        self.solver.addConstr(distance <= gp.quicksum(self.abs_vals[("r2_c1", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)) + gp.quicksum(self.abs_vals[("r1_c2", i, j)] for i in range(n_features) for j in range(self.n_pieces+1)), name=f"xyz")

        self.solver.setObjective(
            distance, gp.GRB.MAXIMIZE
        )

        if self.focus_on_solution:
            try:
                self.solver.setParam("MIPFocus", 1)
            except:
                print(self.solver.Params)
                self.solver.setParam("MIPFocus", 1)
                
        self.solver.update()

        # -- Résolution --
        self.solver.optimize()
        self.status = self.solver.Status
        self.optim_params["status"] = self.status
        try:
            self.optim_params["obj_val"] = self.solver.ObjVal
            self.optim_params["optimality_gap"] = self.solver.MIPGap
        except:
            self.optim_params["obj_val"] = None
            self.optim_params["optimality_gap"] = None
        self.optim_params["runtime"] = self.solver.Runtime
        self.optim_params["num_constr"] = self.solver.NumConstrs
        self.optim_params["num_vars"] = self.solver.NumVars
        self.optim_params["obj_bound"] = self.solver.ObjBound
        self.optim_params["obj_boundC"] = self.solver.ObjBoundC

    def save(self, savedir):
        os.makedirs(savedir, exist_ok=True)

        np.save(os.path.join(savedir, "inflexions.npy"), self.inflexions)
        for uta_model in ["r1", "r2"]:
            all_weights = [[self.marginal_coeffs[uta_model, i, k].x for k in range(self.n_pieces + 1)] for i in range(self.inflexions.shape[0])]
            np.save(os.path.join(savedir, f"{uta_model}_weights.npy"), all_weights)

        with open(os.path.join(savedir, "optim_params.json"), "w") as file:
            json.dump(self.optim_params, file)
