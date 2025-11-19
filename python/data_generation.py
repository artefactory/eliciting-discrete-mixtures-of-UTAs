import numpy as np
import tqdm

from .decision_maker import DecisionMaker

class SyntheticDataGenerator:
    def __init__(
        self,
        n_dms,
        n_criteria,
        mix_decisions=False,
        method_params={},
        noise=0.0,
        gap=0.0,
        decimals=6,
    ):
        self.n_dms = n_dms
        self.n_criteria = n_criteria
        self.mix_decisions = mix_decisions
        self.method_params = method_params
        self.noise = noise  # % of noise = % of pairs that will be reversed
        self.gap = gap
        self.decimals = decimals

        self.instantiate()

    def instantiate(self):
        self.dms = [DecisionMaker(n_criteria=self.n_criteria, n_pieces=self.method_params.get("n_pieces", 5)) for _ in range(self.n_dms)]

        self._marginal_utilities = lambda x: np.array([[dm.get_marginal_utility(criterion_index=i, criterion_value=x[i]) for i in range(len(x))] for dm in self.dms])
        self._utility = lambda x: np.array([dm.get_total_utility(criteria_vector=x) for dm in self.dms])

    def utility(self, X):
        if len(X.shape) == 1:
            return self._utility(X)
        elif len(X.shape) == 2:
            return np.array([self._utility(x) for x in X])
        else:
            raise ValueError("Unsupported shape of X", X.shape)

    def marginal_utilities(self, X):
        if len(X.shape) == 1:
            return self._marginal_utility(X)
        elif len(X.shape) == 2:
            return np.array([self._marginal_utility(x) for x in X])
        else:
            raise ValueError("Unsupported shape of X", X.shape)

    def generate_preferences(
        self, num_pairs, return_utilities=False, return_clusters=False, verbose=0
    ):
        X, Y = [], []

        utilities = [[], []]
        clusters = []
        # Useless now that we have clusters
        populations = [0] * self.n_dms
        if not isinstance(num_pairs, list):
            num_pairs = [np.ceil(num_pairs / self.n_dms)] * self.n_dms
        while len(X) < sum(num_pairs):
            if verbose > 0:
                print(f"{len(X)} events have been created as of now", end="\r")
            
            non_dominance = False
            while not non_dominance:
                x = np.around(
                    np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
                )
                y = np.around(
                    np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
                )
                non_dominance = (np.sum(x-y > 0) != len(x)) & (np.sum(x-y > 0) != 0)

            ux = np.around(self.utility(x), decimals=self.decimals)
            uy = np.around(self.utility(y), decimals=self.decimals)
            if (ux - uy)[np.argmax(ux - uy)] > self.gap:
                if np.sum(ux > uy) == 1 and not self.mix_decisions:
                    if populations[np.argmax(ux > uy)] < num_pairs[np.argmax(ux > uy)]:
                        if np.random.randint(1000) / 1000 >= self.noise:
                            X.append(x)
                            Y.append(y)
                            utilities[0].append(ux)
                            utilities[1].append(uy)
                        else:
                            X.append(y)
                            Y.append(x)
                            utilities[0].append(uy)
                            utilities[1].append(ux)

                        populations[np.argmax(ux - uy)] += 1
                        clusters.append(np.argmax(ux - uy))

                elif np.sum(ux > uy) >= 1 and self.mix_decisions:
                    if populations[np.argmax(ux - uy)] < num_pairs[np.argmax(ux - uy)]:
                        if np.random.randint(1000) / 1000 >= self.noise:
                            X.append(x)
                            Y.append(y)
                            utilities[0].append(ux)
                            utilities[1].append(uy)
                        else:
                            X.append(y)
                            Y.append(x)
                            utilities[0].append(uy)
                            utilities[1].append(ux)

                        populations[np.argmax(ux - uy)] += 1
                        clusters.append(np.argmax(ux - uy))
        if verbose > 0:
            print("Clusters Populations", populations)
        additional_info = {}
        for i in range(self.n_dms):
            additional_info[f"coefficients_{i}"] = self.dms[i].coefficients

        if return_utilities:
            additional_info["utilities_x"] = np.array(utilities)[0]
            additional_info["utilities_y"] = np.array(utilities)[1]
        if return_clusters:
            additional_info["clusters"] = np.array(clusters)
        return np.stack(X), np.stack(Y), additional_info


    def generate_indifferences(
        self, num_pairs, return_utilities=False, return_clusters=False, verbose=0
    ):
        X, Y = [], []

        utilities = [[], []]
        clusters = []
        # Useless now that we have clusters
        populations = [0] * self.n_dms
        if not isinstance(num_pairs, list):
            num_pairs = np.array([np.ceil(num_pairs / self.n_dms)] * self.n_dms).astype(int)

        for i in range(self.n_dms):
            for _ in range(num_pairs[i]):
                if verbose > 0:
                    print(f"{len(X)} events have been created as of now", end="\r")
                x = np.around(
                    np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
                )
                ux = np.around(self.utility(x), decimals=self.decimals)[i]
                y = x.copy()
                
                uyi_1 = None
                while uyi_1 is None:
                    indexes = np.random.permutation(np.arange(len(x)))[:2]

                    uyi_0 = np.around(np.random.uniform(0, 1), decimals=self.decimals)
                    uyi_1 = self.dms[i].get_indifference_on_two_criteria(criterion_i=indexes[0], criterion_j=indexes[1],
                                                                        query_i=x[indexes[0]], p_i=uyi_0, query_j=x[indexes[1]])

                y[indexes[0]] = uyi_0
                y[indexes[1]] = uyi_1

                X.append(x)
                Y.append(y)
                utilities[0].append(ux)
                utilities[1].append(np.around(self.utility(y), decimals=self.decimals)[i])
                populations[i] += 1
                clusters.append(i)
        if verbose > 0:
            print("Clusters Populations", populations)
        additional_info = {}
        for i in range(self.n_dms):
            additional_info[f"coefficients_{i}"] = self.dms[i].coefficients

        if return_utilities:
            additional_info["utilities_x"] = np.array(utilities)[0]
            additional_info["utilities_y"] = np.array(utilities)[1]
        if return_clusters:
            additional_info["clusters"] = np.array(clusters)
        return np.stack(X), np.stack(Y), additional_info


    def generate_indifferences_alldms(
        self, num_pairs, return_utilities=False, return_clusters=False, verbose=0
    ):
        X, Y = [], []

        utilities = [[], []]
        clusters = []
        # Useless now that we have clusters
        populations = [0] * self.n_dms
        if not isinstance(num_pairs, list):
            num_pairs = np.array([np.ceil(num_pairs / self.n_dms)] * self.n_dms).astype(int)

        for _ in tqdm.trange(num_pairs[0]):
            x = np.around(
                np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
            )

            uyj = [None for _ in range(self.n_dms)]
            indexes = np.random.permutation(np.arange(len(x)))[:2]

            count = 0
            while None in uyj:

                uyi_0 = np.random.uniform(0, 1)
                for i in range(self.n_dms):

                    uyj[i] = self.dms[i].get_indifference_on_two_criteria(criterion_i=indexes[0], criterion_j=indexes[1],
                                                                        query_i=x[indexes[0]], p_i=uyi_0, query_j=x[indexes[1]])

                count += 1

            for i in range(self.n_dms):

                y = x.copy()
                y[indexes[0]] = uyi_0
                y[indexes[1]] = uyj[i]

                X.append(x)
                Y.append(y)
                ux = np.around(self.utility(x), decimals=self.decimals)[i]
                utilities[0].append(ux)
                utilities[1].append(np.around(self.utility(y), decimals=self.decimals)[i])
                assert utilities[1][-1] == utilities[0][-1], f"{(np.around(self.utility(y), decimals=self.decimals)[i], )}"
                populations[i] += 1
                clusters.append(i)
        if verbose > 0:
            print("Clusters Populations", populations)
        additional_info = {}
        for i in range(self.n_dms):
            additional_info[f"coefficients_{i}"] = self.dms[i].coefficients

        if return_utilities:
            additional_info["utilities_x"] = np.array(utilities)[0]
            additional_info["utilities_y"] = np.array(utilities)[1]
        if return_clusters:
            additional_info["clusters"] = np.array(clusters)
        return np.stack(X), np.stack(Y), additional_info