import logging
import os
import pickle

import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

from python.data_generation import SyntheticDataGenerator
from python.distances import TwoUTASpaceDiameter


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ### Parameters

    main_save_dir = "/gpfs/workdir/auriauvi/honey"
    n_runs = 2
    n_data = [32, 128, 512, 2048]
    time_limit = 10_800
    inflexions = np.vstack([np.linspace(0, 1., 6)] * 4)
    n_dms = 2
    method_params = {"n_pieces": 5}
    n_criteria = 4
    data_generation_gap = 0.001
    precision_decimals = 3
    indifferences_higher_bound = 1e-5
    preferences_lower_bound = 1e-3
    lipschitz_coefficient = 1e-4

    for run_i in range(n_runs):
        logging.info(f"Starting loop n°{run_i}")
        save_dir = os.path.join(main_save_dir, f"run_b{run_i}")
        os.makedirs(save_dir)
        ### Setup Data Generator
        data_generator = SyntheticDataGenerator(
            n_dms=n_dms,
            n_criteria=n_criteria,
            method_params=method_params,
            gap=data_generation_gap,
            decimals=precision_decimals
        )

        # Generate Data: indifferences & preferences
        X_indiff, Y_indiff, info_indiff = data_generator.generate_indifferences_alldms(num_pairs=np.max(n_data), return_clusters=True, return_utilities=True)

        np.save(os.path.join(save_dir, "X_indiff.npy"), X_indiff)
        np.save(os.path.join(save_dir, "Y_indiff.npy"), Y_indiff)
        with open(os.path.join(save_dir, "info_indiff.pickle"), "wb") as file:
            pickle.dump(info_indiff, file)

        X_pref, Y_pref, info_pref= data_generator.generate_preferences_alldms(num_pairs=np.max(n_data), return_clusters=True, return_utilities=True)
        np.save(os.path.join(save_dir, "X_pref.npy"), X_pref)
        np.save(os.path.join(save_dir, "Y_pref.npy"), Y_pref)
        with open(os.path.join(save_dir, "info_pref.pickle"), "wb") as file:
            pickle.dump(info_pref, file)

        logging.info(f"Data drawn & saved in folder {save_dir}")
        for data_length in n_data:
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir)
            logging.info(f">>> Starting data length {data_length} of loop {run_i}.")

            try:
                with open(os.path.join(model_save_dir, "end.json"), "r") as file:
                    results = json.load(file)
                if not results["training_finished"]:
                    raise ValueError
            except:

                try:
                    with open(os.path.join(model_sav_dir, "indiff_coupled", "fit_params.json"), "r") as file:
                        is_fitted = json.load(file)["optimization_objective"]
                except:
                    dist1 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=indifferences_higher_bound, lipschitz_coeff=lipschitz_coefficient)
                    # dist.solver.setParam("DualReductions", 0)
                    dist1.fit_generic(X_indiff[:data_length],
                                    Y_indiff[:data_length],
                                    time_limit=time_limit,
                                    inflexions=inflexions,
                                    relation_type="indifference",
                                    clustering=np.concatenate([[i, i] for i in range(data_length//2)]))

                    dist1.save(savedir=f"{model_save_dir}/indiff_coupled")
                    logging.info("Model 1 trained & saved")


                try:
                    with open(os.path.join(model_sav_dir, "indiff_singled", "fit_params.json"), "r") as file:
                        is_fitted = json.load(file)["optimization_objective"]
                except:

                    dist2 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=indifferences_higher_bound, lipschitz_coeff=lipschitz_coefficient)
                    # dist.solver.setParam("DualReductions", 0)
                    dist2.fit_generic(X_indiff[:data_length],
                                    Y_indiff[:data_length],
                                    time_limit=time_limit,
                                    inflexions=inflexions,
                                    relation_type="indifference"
                                    )

                    dist2.save(savedir=f"{model_save_dir}/indiff_singled")
                    logging.info("Model 2 trained & saved")


                try:
                    with open(os.path.join(model_sav_dir, "pref_coupled", "fit_params.json"), "r") as file:
                        is_fitted = json.load(file)["optimization_objective"]
                except:
                        
                    dist3 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=preferences_lower_bound, lipschitz_coeff=lipschitz_coefficient)
                    # dist.solver.setParam("DualReductions", 0)
                    dist3.fit_generic(X_pref[:data_length],
                                    Y_pref[:data_length],
                                    time_limit=time_limit,
                                    inflexions=inflexions,
                                    relation_type="preference",
                                    clustering=np.concatenate([[i, i] for i in range(data_length//2)]))

                    dist3.save(savedir=f"{model_save_dir}/pref_coupled")
                    logging.info("Model 3 trained & saved")

                try:
                    with open(os.path.join(model_sav_dir, "pref_singled", "fit_params.json"), "r") as file:
                        is_fitted = json.load(file)["optimization_objective"]
                except:
                    dist4 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=preferences_lower_bound, lipschitz_coeff=lipschitz_coefficient)
                    # dist.solver.setParam("DualReductions", 0)
                    dist4.fit_generic(X_pref[:data_length],
                                    Y_pref[:data_length],
                                    time_limit=time_limit,
                                    inflexions=inflexions,
                                    relation_type="preference"
                                    )

                    dist4.save(savedir=f"{model_save_dir}/pref_singled")
                    logging.info("Model 4 trained & saved")

                with open(os.path.join(model_save_dir, "end.json"), "w") as file:
                    json.dump({"training_finished": True}, file)
