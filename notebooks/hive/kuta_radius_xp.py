import argparse
import json
import logging
import os
import pickle
import time
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

# from python.data_generation import SyntheticDataGenerator
from python.radius import TwoUTASpaceRadius
from python.k_uta import ClusterUTA

from host_config import get_xp_savedir


import numpy as np
import tqdm

from python.decision_maker import DecisionMaker


class OtherSyntheticDataGenerator:
    def __init__(
        self,
        n_dms,
        n_criteria,
        mix_decisions=False,
        method_params={},
        gap=0.0,
        decimals=6,
    ):
        self.n_dms = n_dms
        self.n_criteria = n_criteria
        self.mix_decisions = mix_decisions
        self.method_params = method_params
        # self.noise = noise  # % of noise = % of pairs that will be reversed
        self.gap = gap
        self.decimals = decimals

        self.instantiate()

    def instantiate(self):
        self.dms = [DecisionMaker(n_criteria=self.n_criteria, n_pieces=self.method_params.get("n_pieces", 5), n_decimals=self.decimals) for _ in range(self.n_dms)]

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
            return self._marginal_utilities(X)
        elif len(X.shape) == 2:
            return np.array([self._marginal_utilities(x) for x in X])
        else:
            raise ValueError("Unsupported shape of X", X.shape)

    def generate_preferences(
        self, num_pairs, noise=0., return_utilities=False, return_clusters=False, verbose=0
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
                y = x.copy()

                ### Select two indexes
                swap_indexes = np.random.permutation(np.arange(n_pieces))[:2]
                new_values = np.around(
                    np.random.uniform(0, 1, 2), decimals=self.decimals
                )
                y[swap_indexes] = new_values
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

        count = 0
        pbar = tqdm.tqdm(total = num_pairs[0])
        while count < num_pairs[0]:
        # for _ in tqdm.trange(num_pairs[0]):
            x = np.around(
                np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
            )

            uyj = [None for _ in range(self.n_dms)]
            indexes = np.random.permutation(np.arange(len(x)))[:2]

            subcount = 0
            while None in uyj and subcount < 1_000_000:

                uyi_0 = np.round(np.random.uniform(0, 1), decimals=self.decimals)
                for i in range(self.n_dms):

                    uyj[i] = self.dms[i].get_indifference_on_two_criteria(criterion_i=indexes[0], criterion_j=indexes[1],
                                                                        query_i=x[indexes[0]], p_i=uyi_0, query_j=x[indexes[1]])

                subcount += 1

            if None not in uyj:
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

                    count += 1
                    pbar.update(1)
        # pbar.close()
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

    def generate_preferences_alldms(
        self, num_pairs, noise=0.,return_utilities=False, return_clusters=False, verbose=0
    ):
        X, Y = [], []

        utilities = [[], []]
        clusters = []

        # Useless now that we have clusters
        num_pairs_by_dm = np.ceil(num_pairs / self.n_dms)
        n_data = 0
        while n_data < num_pairs_by_dm:
            if verbose > 0:
                print(f"{len(X)} events have been created as of now", end="\r")
            

            x = np.around(
                np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
            )
            ux = np.around(self.utility(x), decimals=self.decimals)
            for i in range(self.n_dms): 
                added = False
                while not added:
                    non_dominance = False
                    while not non_dominance:
                        y = np.around(
                            np.random.uniform(0, 1, self.n_criteria), decimals=self.decimals
                        )
                        non_dominance = (np.sum(x-y > 0) != len(x)) & (np.sum(x-y > 0) != 0)

                    uy = np.around(self.utility(y), decimals=self.decimals)
                    if (ux - uy)[i] > self.gap:
                        if np.random.randint(1000) / 1000 >= noise:
                            X.append(x)
                            Y.append(y)
                            utilities[0].append(ux)
                            utilities[1].append(uy)
                        else:
                            X.append(y)
                            Y.append(x)
                            utilities[0].append(uy)
                            utilities[1].append(ux)
                        clusters.append(i)
                        added = True

                    elif (uy - ux)[i] > self.gap:
                        if np.random.randint(1000) / 1000 >= noise:
                            X.append(y)
                            Y.append(x)
                            utilities[0].append(uy)
                            utilities[1].append(ux)
                        else:
                            X.append(x)
                            Y.append(y)
                            utilities[0].append(ux)
                            utilities[1].append(uy)
                        clusters.append(i)
                        added = True

                    else:
                        added = False
            n_data += 1
        # if verbose > 0:
        #     print("Clusters Populations", populations)
        additional_info = {}
        for i in range(self.n_dms):
            additional_info[f"coefficients_{i}"] = self.dms[i].coefficients

        if return_utilities:
            additional_info["utilities_x"] = np.array(utilities)[0]
            additional_info["utilities_y"] = np.array(utilities)[1]
        if return_clusters:
            additional_info["clusters"] = np.array(clusters)
        return np.stack(X), np.stack(Y), additional_info


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ### Parameters

    # main_save_dir = "/gpfs/workdir/auriauvi/honey"
    # main_save_dir = "/data/workspace/vincent/elicit"
    main_save_dir = get_xp_savedir()

    n_runs = 10
    base_time_limit = 3_600
    n_criteria = 8
    n_pieces = 5
    inflexions = np.vstack([np.linspace(0, 1., 6)] * n_criteria)

    indiff_n_data = [32, 64, 74, 128, 256, 512]
    indiff_n_data_singled = [32, 64, 74, 128, 256, 512, 2048]
    pref_n_data = [32, 74, 128, 512, 2048, 4096]
    max_n_data = np.max(np.concatenate([indiff_n_data, pref_n_data, indiff_n_data_singled]))
    print("Max number of data:", max_n_data)

    n_dms = 2
    method_params = {"n_pieces": n_pieces}
    data_generation_gap = 0.002
    precision_decimals = 3
    indifferences_higher_bound = 2e-5
    preferences_lower_bound = 2e-3
    lipschitz_coefficient = 1e-5

    n_threads = 15

    parser = argparse.ArgumentParser(
                    prog='Elicit-Radius',
                    description='Synthetic Experiments',
                    epilog='> finished all runs <')
    parser.add_argument("-f", "--flag", default="start", type=str)
    args = parser.parse_args()
    flag = args.flag
    
    for run_i in range(n_runs):
        logging.warning(f"Starting loop n°{run_i}")
        save_dir = os.path.join(main_save_dir, f"run_{flag}{run_i}")

        if os.path.isdir(save_dir):
            X_indiff = np.load(os.path.join(save_dir, "X_indiff.npy"))
            Y_indiff = np.load(os.path.join(save_dir, "Y_indiff.npy"))
		
            X_pref = np.load(os.path.join(save_dir, "X_pref.npy"))
            Y_pref = np.load(os.path.join(save_dir, "Y_pref.npy"))

            with open(os.path.join(save_dir, 'space_center.pkl'), 'rb') as f:
                space_center = pickle.load(f)

        else:
            os.makedirs(save_dir)
            ### Setup Data Generator
            data_generator = OtherSyntheticDataGenerator(
                n_dms=n_dms,
                n_criteria=n_criteria,
                method_params=method_params,
                gap=data_generation_gap,
                decimals=precision_decimals
            )

            # Generate Data: indifferences & preferences
            X_indiff, Y_indiff, info_indiff = data_generator.generate_indifferences_alldms(num_pairs=max_n_data, return_clusters=True, return_utilities=True)

            np.save(os.path.join(save_dir, "X_indiff.npy"), X_indiff)
            np.save(os.path.join(save_dir, "Y_indiff.npy"), Y_indiff)
            with open(os.path.join(save_dir, "info_indiff.pickle"), "wb") as file:
                pickle.dump(info_indiff, file)

            X_pref, Y_pref, info_pref= data_generator.generate_preferences_alldms(num_pairs=2*max_n_data, return_clusters=True, return_utilities=True)
            np.save(os.path.join(save_dir, "X_pref.npy"), X_pref)
            np.save(os.path.join(save_dir, "Y_pref.npy"), Y_pref)
            with open(os.path.join(save_dir, "info_pref.pickle"), "wb") as file:
                pickle.dump(info_pref, file)

            space_center = {}
            for dm in range(n_dms):
                for i in range(n_criteria):
                    for j in range(n_pieces+1):
                        space_center[(f"c{dm+1}", i, j)] = info_pref[f"coefficients_{dm}"][i][j]
            
            print(space_center)
            with open(os.path.join(save_dir, 'space_center.pkl'), 'wb') as f:
                pickle.dump(space_center, f)

            logging.warning(f"Data drawn & saved in folder {save_dir}")

        for data_length in indiff_n_data:
            print("Start Indifferences n_data:", data_length)
            time_limit = base_time_limit
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)
            logging.warning(f">>> Starting data length {data_length} of loop {run_i}.")


            try:
                with open(os.path.join(model_save_dir, "indiff_coupled", "kuta.pkl"), "rb") as file:
                    is_fitted = pickle.load(file)
            except FileNotFoundError:
                
                kuta = ClusterUTA(
                    n_clusters=n_dms,
                    n_pieces=n_pieces,
                    epsilon=1e-3,
                )
                hist = kuta.fit_minimize_indiff_gap(
                    X_indiff[:data_length],
                    Y_indiff[:data_length], 
                    unclustering=np.concatenate([[i, i] for i in range(data_length//2)]),
                    time_limit=time_limit
                    )

                kuta_dict = {}
                for i in range(n_dms):
                    for crit in range(n_criteria):
                        for k in range(6):
                            kuta_dict[i, crit, k] = kuta.marginal_coeffs[i, crit, k].x
                dist_1 = 0
                for crit in range(n_criteria):
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(6)]) - np.array([space_center["c1", crit, k] for k in range(6)])))
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(6)]) - np.array([space_center["c2", crit, k] for k in range(6)])))

                dist_2 = 0
                for crit in range(n_criteria):
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(6)]) - np.array([space_center["c2", crit, k] for k in range(6)])))
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(6)]) - np.array([space_center["c1", crit, k] for k in range(6)])))

                dist = np.min([dist_1, dist_2])
                kuta_dict["dist"] = dist
                os.makedirs(f"{model_save_dir}/indiff_coupled", exist_ok=True)
                with open(os.path.join(f"{model_save_dir}/indiff_coupled", "kuta.pkl"), "wb") as file:
                    pickle.dump(kuta_dict, file)
                logging.warning(f"Model 2 trained & saved in {model_save_dir}/indiff_coupled")

        for data_length in indiff_n_data_singled:
            print("Start Indifferences n_data:", data_length)
            time_limit = base_time_limit
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)

            try:
                with open(os.path.join(model_save_dir, "indiff_singled", "kuta.pkl"), "rb") as file:
                    is_fitted = pickle.load(file)
            except FileNotFoundError:
                kuta = ClusterUTA(
                    n_clusters=n_dms,
                    n_pieces=n_pieces,
                    epsilon=1e-3,
                )
                hist = kuta.fit_minimize_indiff_gap(
                    X_indiff[:data_length],
                    Y_indiff[:data_length], 
                    time_limit=time_limit
                    )

                kuta_dict = {}
                for i in range(n_dms):
                    for crit in range(n_criteria):
                        for k in range(6):
                            kuta_dict[i, crit, k] = kuta.marginal_coeffs[i, crit, k].x
                dist_1 = 0
                for crit in range(n_criteria):
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(6)]) - np.array([space_center["c1", crit, k] for k in range(6)])))
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(6)]) - np.array([space_center["c2", crit, k] for k in range(6)])))

                dist_2 = 0
                for crit in range(n_criteria):
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(6)]) - np.array([space_center["c2", crit, k] for k in range(6)])))
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(6)]) - np.array([space_center["c1", crit, k] for k in range(6)])))

                dist = np.min([dist_1, dist_2])
                kuta_dict["dist"] = dist
                os.makedirs(f"{model_save_dir}/indiff_singled", exist_ok=True)
                with open(os.path.join(f"{model_save_dir}/indiff_singled", "kuta.pkl"), "wb") as file:
                    pickle.dump(kuta_dict, file)
                logging.warning(f"Model 2 trained & saved in {model_save_dir}/indiff_singled")


        for data_length in pref_n_data:
            print("Start Preferences n_data:", data_length)

            time_limit = base_time_limit
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)
            try:
                with open(os.path.join(model_save_dir, "pref_coupled", "kuta.pkl"), "rb") as file:
                    is_fitted = pickle.load(file)
            except FileNotFoundError:

                kuta = ClusterUTA(
                    n_clusters=n_dms,
                    n_pieces=n_pieces,
                    epsilon=1e-3,
                )
                hist = kuta.fit_maximize_gap(
                    X_pref[:data_length],
                    Y_pref[:data_length], 
                    unclustering=np.concatenate([[i, i] for i in range(data_length//2)]),
                    time_limit=time_limit
                    )

                kuta_dict = {}
                for i in range(n_dms):
                    for crit in range(n_criteria):
                        for k in range(6):
                            kuta_dict[i, crit, k] = kuta.marginal_coeffs[i, crit, k].x
                dist_1 = 0
                for crit in range(n_criteria):
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(n_pieces+1)]) - np.array([space_center["c1", crit, k] for k in range(n_pieces+1)])))
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(n_pieces+1)]) - np.array([space_center["c2", crit, k] for k in range(n_pieces+1)])))

                dist_2 = 0
                for crit in range(n_criteria):
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(n_pieces+1)]) - np.array([space_center["c2", crit, k] for k in range(n_pieces+1)])))
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(n_pieces+1)]) - np.array([space_center["c1", crit, k] for k in range(n_pieces+1)])))

                dist = np.min([dist_1, dist_2])
                kuta_dict["dist"] = dist
                os.makedirs(f"{model_save_dir}/pref_coupled", exist_ok=True)
                with open(os.path.join(f"{model_save_dir}/pref_coupled", "kuta.pkl"), "wb") as file:
                    pickle.dump(kuta_dict, file)
                print(model_save_dir, "kuta saved")
               

            try:
                with open(os.path.join(model_save_dir, "pref_singled", "kuta.pkl"), "rb") as file:
                    is_fitted = pickle.load(file)
            except FileNotFoundError:

                kuta = ClusterUTA(
                    n_clusters=n_dms,
                    n_pieces=n_pieces,
                    epsilon=1e-3,
                )
                hist = kuta.fit_maximize_gap(X_pref[:data_length], Y_pref[:data_length], 
                    time_limit=time_limit)

                kuta_dict = {}
                for i in range(n_dms):
                    for crit in range(n_criteria):
                        for k in range(6):
                            kuta_dict[i, crit, k] = kuta.marginal_coeffs[i, crit, k].x
                dist_1 = 0
                for crit in range(n_criteria):
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(6)]) - np.array([space_center["c1", crit, k] for k in range(6)])))
                    dist_1 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(6)]) - np.array([space_center["c2", crit, k] for k in range(6)])))

                dist_2 = 0
                for crit in range(n_criteria):
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[0, crit, k].x for k in range(6)]) - np.array([space_center["c2", crit, k] for k in range(6)])))
                    dist_2 += np.sum(np.abs(np.array([kuta.marginal_coeffs[1, crit, k].x for k in range(6)]) - np.array([space_center["c1", crit, k] for k in range(6)])))

                dist = np.min([dist_1, dist_2])
                kuta_dict["dist"] = dist
                os.makedirs(f"{model_save_dir}/pref_singled", exist_ok=True)
                with open(os.path.join(f"{model_save_dir}/pref_singled", "kuta.pkl"), "wb") as file:
                    pickle.dump(kuta_dict, file)
