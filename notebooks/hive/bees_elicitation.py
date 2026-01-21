import argparse
import json
import logging
import os
import pickle

import sys
sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np

from python.data_generation import SyntheticDataGenerator
from python.distances import TwoUTASpaceDiameter

from host_config import get_xp_savedir


logger = logging.getLogger(__name__)

if __name__ == "__main__":

    ### Parameters

    # main_save_dir = "/gpfs/workdir/auriauvi/honey"
    # main_save_dir = "/data/workspace/vincent/elicit"
    main_save_dir = get_xp_savedir()
    n_runs = 10
    indiff_n_data = [32, 64, 74, 128, 256]
    pref_n_data = [32, 74, 128, 512, 2048, 4096]
    base_time_limit = 10_800
    inflexions = np.vstack([np.linspace(0, 1., 6)] * 4)
    n_dms = 2
    method_params = {"n_pieces": 5}
    n_criteria = 4
    data_generation_gap = 0.002
    precision_decimals = 3
    indifferences_higher_bound = 2e-5
    preferences_lower_bound = 2e-3
    lipschitz_coefficient = 1e-5

    n_threads = 16

    parser = argparse.ArgumentParser(
                    prog='Elicit-World',
                    description='Synthetic Experiments',
                    epilog='> finished all runs <')
    parser.add_argument("-f", "--flag", default="z", type=str)
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

            # if len(X_indiff) < 2*np.max(np.concatenate([indiff_n_data, pref_n_data])) or len(X_pref) < 2*np.max(np.concatenate([indiff_n_data, pref_n_data])):
            #     logging.warning(f"Data not found in folder {save_dir}, regenerating data. found {len(X_indiff)} indifferences and {len(X_pref)} preferences.")
            #     data_generator = SyntheticDataGenerator(
            #         n_dms=n_dms,
            #         n_criteria=n_criteria,
            #         method_params=method_params,
            #         gap=data_generation_gap,
            #         decimals=precision_decimals
            #     )
            #     with open(os.path.join(save_dir, "info_indiff.pickle"), "rb") as file:
            #         info_indiff = pickle.load(file)
            #     with open(os.path.join(save_dir, "info_pref.pickle"), "rb") as file:
            #         info_pref = pickle.load(file)
            #     data_generator.dms[0].load_from_parameters(info_indiff["coefficients_0"])
            #     data_generator.dms[1].load_from_parameters(info_indiff["coefficients_1"])

            #     X_indiff_2, Y_indiff_2, info_indiff_2 = data_generator.generate_indifferences_alldms(num_pairs=2*np.max(np.concatenate([indiff_n_data, pref_n_data])), return_clusters=True, return_utilities=True)
            #     X_indiff = np.concatenate([X_indiff, X_indiff_2], axis=0)
            #     Y_indiff = np.concatenate([Y_indiff, Y_indiff_2], axis=0)

            #     assert (info_indiff_2["coefficients_0"] == info_indiff["coefficients_0"]).all()
            #     for key in ['utilities_x', 'utilities_y', 'clusters']:
            #         info_indiff[key] = np.concatenate([info_indiff[key], info_indiff_2[key]], axis=0)

                
            #     np.save(os.path.join(save_dir, "X_indiff.npy"), X_indiff)
            #     np.save(os.path.join(save_dir, "Y_indiff.npy"), Y_indiff)
            #     with open(os.path.join(save_dir, "info_indiff.pickle"), "wb") as file:
            #         pickle.dump(info_indiff, file)

            #     X_pref_2, Y_pref_2, info_pref_2 = data_generator.generate_preferences_alldms(num_pairs=2*np.max(np.concatenate([indiff_n_data, pref_n_data])), return_clusters=True, return_utilities=True)
                
            #     assert (info_pref_2["coefficients_0"] == info_pref["coefficients_0"]).all()
            #     X_pref = np.concatenate([X_pref, X_pref_2], axis=0)
            #     Y_pref = np.concatenate([Y_pref, Y_pref_2], axis=0)
            #     for key in ['utilities_x', 'utilities_y', 'clusters']:
            #         info_pref[key] = np.concatenate([info_pref[key], info_pref_2[key]], axis=0)
                
            #     np.save(os.path.join(save_dir, "X_pref.npy"), X_pref)
            #     np.save(os.path.join(save_dir, "Y_pref.npy"), Y_pref)
            #     with open(os.path.join(save_dir, "info_pref.pickle"), "wb") as file:
            #         pickle.dump(info_pref, file)

        else:
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
            X_indiff, Y_indiff, info_indiff = data_generator.generate_indifferences_alldms(num_pairs=2*np.max(np.concatenate([indiff_n_data, pref_n_data])), return_clusters=True, return_utilities=True)

            np.save(os.path.join(save_dir, "X_indiff.npy"), X_indiff)
            np.save(os.path.join(save_dir, "Y_indiff.npy"), Y_indiff)
            with open(os.path.join(save_dir, "info_indiff.pickle"), "wb") as file:
                pickle.dump(info_indiff, file)

            X_pref, Y_pref, info_pref= data_generator.generate_preferences_alldms(num_pairs=2*np.max(np.concatenate([indiff_n_data, pref_n_data])), return_clusters=True, return_utilities=True)
            np.save(os.path.join(save_dir, "X_pref.npy"), X_pref)
            np.save(os.path.join(save_dir, "Y_pref.npy"), Y_pref)
            with open(os.path.join(save_dir, "info_pref.pickle"), "wb") as file:
                pickle.dump(info_pref, file)

            logging.warning(f"Data drawn & saved in folder {save_dir}")
        for data_length in indiff_n_data:
            time_limit = base_time_limit + base_time_limit * np.array(data_length >= 2048).astype(int)
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)
            logging.warning(f">>> Starting data length {data_length} of loop {run_i}.")

            try:
                with open(os.path.join(model_save_dir, "end.json"), "r") as file:
                    results = json.load(file)
                if not results["training_finished"]:
                    raise ValueError
            except:

                try:
                    with open(os.path.join(model_save_dir, "indiff_coupled", "fit_params.json"), "r") as file:
                        is_fitted = json.load(file)["optimization_objective"]
                except:
                    dist1 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=indifferences_higher_bound, lipschitz_coeff=lipschitz_coefficient)
                    # dist.solver.setParam("DualReductions", 0)
                    dist1.fit_generic(X_indiff[:data_length],
                                    Y_indiff[:data_length],
                                    time_limit=time_limit,
                                    n_threads=n_threads,
                                    inflexions=inflexions,
                                    relation_type="indifference",
                                    clustering=np.concatenate([[i, i] for i in range(data_length//2)]))
                    try:
                        dist1.save(savedir=f"{model_save_dir}/indiff_coupled")
                    except:
                        logging.error(dist1.solver.Status)
                        dist1.save(savedir=f"{model_save_dir}/indiff_coupled")
                    logging.warning(f"Model 1 trained & saved in {model_save_dir}/indiff_coupled")


                try:
                    with open(os.path.join(model_save_dir, "indiff_singled", "fit_params.json"), "r") as file:
                        is_fitted = json.load(file)["optimization_objective"]
                except:

                    dist2 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=indifferences_higher_bound, lipschitz_coeff=lipschitz_coefficient)
                    # dist.solver.setParam("DualReductions", 0)
                    dist2.fit_generic(X_indiff[:data_length],
                                    Y_indiff[:data_length],
                                    time_limit=time_limit,
                                    n_threads=n_threads,
                                    inflexions=inflexions,
                                    relation_type="indifference",
                                    warm_zs=True,
                                    )
                    try:
                        dist2.save(savedir=f"{model_save_dir}/indiff_singled")
                    except:
                        logging.error(dist2.solver.Status)
                        dist2.save(savedir=f"{model_save_dir}/indiff_singled")
                    logging.warning(f"Model 2 trained & saved in {model_save_dir}/indiff_singled")


        for data_length in pref_n_data:
            print("Start Preferences n_data:", data_length)
            time_limit = base_time_limit + base_time_limit * np.array(data_length >= 512).astype(int) + base_time_limit * np.array(data_length >= 4096).astype(int)
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)
            try:
                with open(os.path.join(model_save_dir, "pref_coupled", "fit_params.json"), "r") as file:
                    is_fitted = json.load(file)["optimization_objective"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                    
                dist3 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=preferences_lower_bound, lipschitz_coeff=lipschitz_coefficient)
                # dist.solver.setParam("DualReductions", 0)
                dist3.fit_generic(X_pref[:data_length],
                                Y_pref[:data_length],
                                time_limit=time_limit,
                                n_threads=n_threads,
                                inflexions=inflexions,
                                relation_type="preference", warm_zs=True,
                                clustering=np.concatenate([[i, i] for i in range(data_length//2)]))

                try:
                    dist3.save(savedir=f"{model_save_dir}/pref_coupled")
                except:
                    logging.error(dist3.solver.status)
                    dist3.save(savedir=f"{model_save_dir}/pref_coupled")
                logging.warning(f"Model 3 trained & saved in {model_save_dir}/pref_coupled")

            try:
                with open(os.path.join(model_save_dir, "pref_singled", "fit_params.json"), "r") as file:
                    is_fitted = json.load(file)["optimization_objective"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                dist4 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=preferences_lower_bound, lipschitz_coeff=lipschitz_coefficient)
                # dist.solver.setParam("DualReductions", 0)
                dist4.fit_generic(X_pref[:data_length],
                                Y_pref[:data_length],
                                time_limit=time_limit,
                                n_threads=n_threads,
                                inflexions=inflexions,
                                relation_type="preference", warm_zs=True
                                )
                try:
                    dist4.save(savedir=f"{model_save_dir}/pref_singled")
                except:
                    logging.error(dist4.solver.Status)
                    dist4.save(savedir=f"{model_save_dir}/pref_singled")
                logging.warning(f"Model 4 trained & saved in {model_save_dir}/pref_singled")

            with open(os.path.join(model_save_dir, "end.json"), "w") as file:
                json.dump({"training_finished": True}, file)
