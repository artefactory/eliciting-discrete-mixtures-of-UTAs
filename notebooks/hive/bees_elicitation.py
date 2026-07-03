import argparse
import json
import logging
import os
import pickle
import time

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
    indiff_n_data_singled = [32, 64, 74, 128, 256, 512, 2048]
    pref_n_data = [32, 74, 128, 512, 2048, 4096]
    base_time_limit = 10_800 * 2
    inflexions = np.vstack([np.linspace(0, 1., 6)] * 4)
    n_dms = 2
    method_params = {"n_pieces": 5}
    n_criteria = 4
    data_generation_gap = 0.002
    precision_decimals = 3
    indifferences_higher_bound = 2e-5
    preferences_lower_bound = 2e-3
    lipschitz_coefficient = 1e-5

    n_threads = 35

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
            print("Start Indifferences n_data:", data_length)
            # time_limit = base_time_limit + base_time_limit * np.array(data_length >= 2048).astype(int)
            time_limit = base_time_limit
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)
            logging.warning(f">>> Starting data length {data_length} of loop {run_i}.")


            try:
                with open(os.path.join(model_save_dir, "indiff_coupled", "optim_params.json"), "r") as file:
                    is_fitted = json.load(file)["obj_val"]
            except FileNotFoundError:
                dist1 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=indifferences_higher_bound, lipschitz_coeff=lipschitz_coefficient, focus_on_solution=True)
                # dist.solver.setParam("DualReductions", 0)
                t0 = time.time()
                dist1.fit_generic(X_indiff[:data_length],
                                Y_indiff[:data_length],
                                time_limit=time_limit,
                                n_threads=n_threads,
                                inflexions=inflexions,
                                relation_type="indifference",
                                clustering=np.concatenate([[i, i] for i in range(data_length//2)]))
                t1 = time.time()
                try:
                    dist1.save(savedir=f"{model_save_dir}/indiff_coupled")

                    with open(os.path.join(f"{model_save_dir}/indiff_coupled", "optim_params.json"), "r") as file:
                        opt_file = json.load(file)
                    opt_file["optimization_time"] = t1 - t0
                    with open(os.path.join(f"{model_save_dir}/indiff_coupled", "optim_params.json"), "w") as file:
                        json.dump(opt_file, file)

                except:
                    logging.error(dist1.solver.Status)
                    with open(f"{model_save_dir}/indiff_coupled/not_found.txt", "w") as file:
                        file.write(f"{dist1.solver.ObjBound} - {dist1.solver.ObjBoundC}")
                    # dist1.save(savedir=f"{model_save_dir}/indiff_coupled")
                logging.warning(f"Model 1 trained & saved in {model_save_dir}/indiff_coupled")

        for data_length in indiff_n_data_singled:
            print("Start Indifferences n_data:", data_length)
            # time_limit = base_time_limit + base_time_limit * np.array(data_length >= 512).astype(int)
            time_limit = base_time_limit
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)

            try:
                with open(os.path.join(model_save_dir, "indiff_singled", "optim_params.json"), "r") as file:
                    is_fitted = json.load(file)["obj_val"]
            except FileNotFoundError:
                dist2 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=indifferences_higher_bound, lipschitz_coeff=lipschitz_coefficient, focus_on_solution=True)
                # dist.solver.setParam("DualReductions", 0)
                t0 = time.time()
                dist2.fit_generic(X_indiff[:data_length],
                                Y_indiff[:data_length],
                                time_limit=time_limit,
                                n_threads=n_threads,
                                inflexions=inflexions,
                                relation_type="indifference",
                                warm_zs=True,
                                )
                t1 = time.time()
                try:
                    dist2.save(savedir=f"{model_save_dir}/indiff_singled")

                    with open(os.path.join(f"{model_save_dir}/indiff_singled", "optim_params.json"), "r") as file:
                        opt_file = json.load(file)
                    opt_file["optimization_time"] = t1 - t0
                    with open(os.path.join(f"{model_save_dir}/indiff_singled", "optim_params.json"), "w") as file:
                        json.dump(opt_file, file)

                except:
                    logging.error(dist2.solver.Status)
                    with open(f"{model_save_dir}/indiff_singled/not_found.txt", "w") as file:
                        file.write(f"{dist2.solver.ObjBound} - {dist2.solver.ObjBoundC}")

                    # if data_length >= 512:
                    # if data_length >= 1e6:
                    #     with open(os.path.join(f"{model_save_dir}/indiff_singled", "optim_params.json"), "w") as file:
                    #             json.dump({"optimization_status": dist2.solver.Status, "obj_val": 1e-3, "optimization_time": t1-t0}, file)

                    # else:
                    #     dist2.save(savedir=f"{model_save_dir}/indiff_singled")

                    #     with open(os.path.join(f"{model_save_dir}/indiff_singled", "optim_params.json"), "r") as file:
                    #         opt_file = json.load(file)
                    #     opt_file["optimization_time"] = t1 - t0
                    #     with open(os.path.join(f"{model_save_dir}/indiff_singled", "optim_params.json"), "w") as file:
                    #         json.dump(opt_file, file)

                logging.warning(f"Model 2 trained & saved in {model_save_dir}/indiff_singled")


        for data_length in pref_n_data:
            print("Start Preferences n_data:", data_length)
            # time_limit = base_time_limit + base_time_limit * np.array(data_length >= 512).astype(int) + base_time_limit * np.array(data_length >= 4096).astype(int)
            time_limit = base_time_limit
            model_save_dir = os.path.join(save_dir, f"ndata_{data_length}")
            os.makedirs(model_save_dir, exist_ok=True)
            try:
                with open(os.path.join(model_save_dir, "pref_coupled", "optim_params.json"), "r") as file:
                    is_fitted = json.load(file)["obj_val"]
            except FileNotFoundError:
                    
                dist3 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=preferences_lower_bound, lipschitz_coeff=lipschitz_coefficient, focus_on_solution=True)
                # dist.solver.setParam("DualReductions", 0)
                t0 = time.time()
                dist3.fit_generic(X_pref[:data_length],
                                Y_pref[:data_length],
                                time_limit=time_limit,
                                n_threads=n_threads,
                                inflexions=inflexions,
                                relation_type="preference", warm_zs=True,
                                clustering=np.concatenate([[i, i] for i in range(data_length//2)]))
                t1 = time.time()

                try:
                    dist3.save(savedir=f"{model_save_dir}/pref_coupled")

                    with open(os.path.join(f"{model_save_dir}/pref_coupled", "optim_params.json"), "r") as file:
                        opt_file = json.load(file)
                    opt_file["optimization_time"] = t1 - t0
                    with open(os.path.join(f"{model_save_dir}/pref_coupled", "optim_params.json"), "w") as file:
                        json.dump(opt_file, file)

                    logging.warning(f"Model 3 trained & saved in {model_save_dir}/pref_coupled")
                except:
                    logging.error(dist3.solver.status)

                    with open(f"{model_save_dir}/pref_coupled/not_found.txt", "w") as file:
                        file.write(f"{dist3.solver.ObjBound} - {dist3.solver.ObjBoundC}")
                    # if data_length >= 4096:
                    # if data_length >= 1e6:
                    #     with open(os.path.join(f"{model_save_dir}/pref_coupled", "optim_params.json"), "w") as file:
                    #             json.dump({"optimization_status": dist3.solver.Status, "obj_val": 1e-5, "optimization_time": t1 - t0}, file)

                    # else:
                    #     dist3.save(savedir=f"{model_save_dir}/pref_coupled")

                    #     with open(os.path.join(f"{model_save_dir}/pref_coupled", "optim_params.json"), "r") as file:
                    #         opt_file = json.load(file)
                    #     opt_file["optimization_time"] = t1 - t0
                    #     with open(os.path.join(f"{model_save_dir}/pref_coupled", "optim_params.json"), "w") as file:
                    #         json.dump(opt_file, file)

            try:
                with open(os.path.join(model_save_dir, "pref_singled", "optim_params.json"), "r") as file:
                    is_fitted = json.load(file)["obj_val"]
            except FileNotFoundError:
                dist4 = TwoUTASpaceDiameter(n_pieces=method_params.get("n_pieces", 5), epsilon=preferences_lower_bound, lipschitz_coeff=lipschitz_coefficient, focus_on_solution=True)
                # dist.solver.setParam("DualReductions", 0)
                time_limit = base_time_limit
                t0 = time.time()
                dist4.fit_generic(X_pref[:data_length],
                                Y_pref[:data_length],
                                time_limit=time_limit,
                                n_threads=n_threads,
                                inflexions=inflexions,
                                relation_type="preference", warm_zs=True
                                )
                t1 = time.time()
                try:
                    dist4.save(savedir=f"{model_save_dir}/pref_singled")

                    with open(os.path.join(f"{model_save_dir}/pref_singled", "optim_params.json"), "r") as file:
                        opt_file = json.load(file)
                    opt_file["optimization_time"] = t1 - t0
                    with open(os.path.join(f"{model_save_dir}/pref_singled", "optim_params.json"), "w") as file:
                        json.dump(opt_file, file)
                except:
                    logging.error(dist4.solver.Status)

                    with open(f"{model_save_dir}/pref_singled/not_found.txt", "w") as file:
                        file.write(f"{dist4.solver.ObjBound} - {dist4.solver.ObjBoundC}")

                    # if data_length >= 4096:
                    # if data_length > 1e6:
                    #     with open(os.path.join(f"{model_save_dir}/pref_singled", "optim_params.json"), "w") as file:
                    #             json.dump({"optimization_status": dist4.solver.Status, "obj_val": 1e-5, "optimization_time": t1 - t0}, file)
                    # else:
                    #     dist4.save(savedir=f"{model_save_dir}/pref_singled")
                        
                    #     with open(os.path.join(f"{model_save_dir}/pref_singled", "optim_params.json"), "r") as file:
                    #         opt_file = json.load(file)
                    #     opt_file["optimization_time"] = t1 - t0
                    #     with open(os.path.join(f"{model_save_dir}/pref_singled", "optim_params.json"), "w") as file:
                    #         json.dump(opt_file, file)
                logging.warning(f"Model 4 trained & saved in {model_save_dir}/pref_singled")
