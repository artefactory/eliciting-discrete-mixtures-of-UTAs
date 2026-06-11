import pickle
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np

from python.data_generation import SyntheticDataGenerator
from python.distances import TwoUTASpaceDiameter

if __name__ == "__main__":
    # generator = SyntheticDataGenerator(
    #     n_dms=2,
    #     n_criteria=4,
    #     method_params={"n_pieces": 5},
    #     gap=0.001,
    #     decimals=3
    # )
    # X_indiff, Y_indiff, info_indiff = generator.generate_indifferences_alldms(num_pairs=1_000, return_clusters=True, return_utilities=True)
    # np.save("xp_data/X_indiff.npy", X_indiff)
    # np.save("xp_data/Y_indiff.npy", Y_indiff)
    # with open("xp_data/info_indiff.pickle", "wb") as file:
    #     pickle.dump(info_indiff, file)
    # X_pref, Y_pref, info_pref= generator.generate_preferences_all_dms(num_pairs=1_000, return_clusters=True, return_utilities=True)
    X_pref = np.load("xp_data/X_pref.npy")
    Y_pref = np.load("xp_data/Y_pref.npy")
    # with open("xp_data/info_pref.pickle", "wb") as file:
    #     pickle.dump(info_pref, file)

    for length in [1_000]:

        # dist = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3)
        # # dist.solver.setParam("DualReductions", 0)
        # dist.fit_from_coupled_indifferences(X_indiff[:length], Y_indiff[:length], time_limit=10_800, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4), warm_zs=True)

        # print(f"ObjVal for length {length}:", dist.solver.ObjVal, "Optimization Status:", dist.solver.Status)
        # plt.figure(figsize=(12, 16))
        # for i in range(4):
        #     plt.subplot(4, 2, i+1)
        #     plt.plot([dist.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
        #     plt.plot([dist.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
        #     plt.plot([dist.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
        #     plt.plot([dist.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        # for i in range(4):
        #     plt.subplot(4, 2, i+5)
        #     plt.plot(generator.dms[0].coefficients[i], c="blue")
        #     plt.plot(generator.dms[1].coefficients[i], c="orange")
        # plt.suptitle(dist.solver.ObjVal)
        # plt.savefig(f"xp_data/indiff_{length}_{dist.solver.Status}_couple.png")
        # plt.show()


        # dist2 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-4, lipschitz_coeff=2e-4)
        # # dist.solver.setParam("DualReductions", 0)
        # dist2.fit_from_indifferences(X_indiff[:length], Y_indiff[:length], time_limit=10_800, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4))

        # print(f"ObjVal for length {length}:", dist2.solver.ObjVal, "Optimization Status:", dist2.solver.Status)
        # plt.figure(figsize=(12, 16))
        # for i in range(4):
        #     plt.subplot(4, 2, i+1)
        #     plt.plot([dist2.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
        #     plt.plot([dist2.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
        #     plt.plot([dist2.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
        #     plt.plot([dist2.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        # for i in range(4):
        #     plt.subplot(4, 2, i+5)
        #     plt.plot(generator.dms[0].coefficients[i], c="blue")
        #     plt.plot(generator.dms[1].coefficients[i], c="orange")
        # plt.suptitle(dist2.solver.ObjVal)
        # plt.savefig(f"xp_data/indiff_{length}_{dist2.solver.Status}_single.png")
        # plt.show()

        # dist3 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3)
        # # dist.solver.setParam("DualReductions", 0)
        # dist3.fit_from_coupled_preferences(X_pref[:length], Y_pref[:length], time_limit=10_800, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4))

        # print(f"ObjVal for length {length}:", dist3.solver.ObjVal, "Optimization Status:", dist3.solver.Status)
        # plt.figure(figsize=(12, 16))
        # for i in range(4):
        #     plt.subplot(4, 2, i+1)
        #     plt.plot([dist3.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
        #     plt.plot([dist3.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
        #     plt.plot([dist3.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
        #     plt.plot([dist3.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        # for i in range(4):
        #     plt.subplot(4, 2, i+5)
        #     plt.plot(generator.dms[0].coefficients[i], c="blue")
        #     plt.plot(generator.dms[1].coefficients[i], c="orange")
        # plt.suptitle(dist3.solver.ObjVal)
        # plt.savefig(f"xp_data/pref_{length}_{dist3.solver.Status}_couple.png")
        # plt.show()

        dist4 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3)
        dist4.solver.setParam("Threads", 24)
        dist4.fit(X_pref[:length], Y_pref[:length], time_limit=7200, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4))

        print(f"ObjVal for length {length}:", dist4.solver.ObjVal, "Optimization Status:", dist4.solver.Status)
        plt.figure(figsize=(12, 16))
        for i in range(4):
            plt.subplot(4, 2, i+1)
            plt.plot([dist4.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
            plt.plot([dist4.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
            plt.plot([dist4.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
            plt.plot([dist4.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        # for i in range(4):
        #     plt.subplot(4, 2, i+5)
        #     plt.plot(generator.dms[0].coefficients[i], c="blue")
        #     plt.plot(generator.dms[1].coefficients[i], c="orange")
        plt.suptitle(dist4.solver.ObjVal)
        plt.savefig(f"xp_data/pref_{length}_{dist4.solver.Status}_single.png")
        plt.show()