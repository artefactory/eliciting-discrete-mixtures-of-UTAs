import pickle
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np

from python.data_generation import SyntheticDataGenerator
from python.distances import TwoUTASpaceDiameter

if __name__ == "__main__":
    generator = SyntheticDataGenerator(
        n_dms=2,
        n_criteria=4,
        method_params={"n_pieces": 5},
        gap=0.001,
        decimals=3
    )
    X_indiff, Y_indiff, info_indiff = generator.generate_indifferences_alldms(num_pairs=1_000, return_clusters=True, return_utilities=True)
    np.save("xp_3/X_indiff.npy", X_indiff)
    np.save("xp_3/Y_indiff.npy", Y_indiff)
    with open("xp_data/info_indiff.pickle", "wb") as file:
        pickle.dump(info_indiff, file)
    X_pref, Y_pref, info_pref= generator.generate_preferences_alldms(num_pairs=1_000, return_clusters=True, return_utilities=True)
    np.save("xp_3/X_pref.npy", X_pref)
    np.save("xp_3/Y_pref.npy", Y_pref)
    with open("xp_3/info_pref.pickle", "wb") as file:
        pickle.dump(info_pref, file)

    for length in [40, 80, 400, 800]:

        dist = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-5, lipschitz_coeff=1e-4)
        # dist.solver.setParam("DualReductions", 0)
        dist.fit_generic(X_indiff[:length],
                        Y_indiff[:length],
                        time_limit=10_800,
                        n_threads=16,
                        inflexions=np.vstack([np.linspace(0, 1., 6)] * 4),
                        relation_type="indifference",
                        clustering=np.concatenate([[i, i] for i in range(length//2)]))

        print(f"ObjVal for length {length}:", dist.solver.ObjVal, "Optimization Status:", dist.solver.Status)
        plt.figure(figsize=(12, 16))
        for i in range(4):
            plt.subplot(4, 2, i+1)
            plt.plot([dist.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
            plt.plot([dist.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
            plt.plot([dist.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
            plt.plot([dist.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        for i in range(4):
            plt.subplot(4, 2, i+5)
            plt.plot(generator.dms[0].coefficients[i], c="blue")
            plt.plot(generator.dms[1].coefficients[i], c="orange")
        plt.suptitle(dist.solver.ObjVal)
        plt.savefig(f"xp_3/indiff_{length}_{dist.solver.Status}_couple.png")
        plt.show()

        coefficients = {
            "s1": [],
            "s2": [],
            "d1": [],
            "d2": [],
        }
        for name in coefficients.keys():
            for i in range(4):
                coefficients[name].append([dist.marginal_coeffs[name, i, k].x for k in range(6)])
        np.save(f"xp_3/indiff_{length}_{dist.solver.Status}_couple.npy", np.stack([list(coefficients.values())]))


        dist2 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-5, lipschitz_coeff=1e-4)
        # dist.solver.setParam("DualReductions", 0)
        dist2.fit_generic(X_indiff[:length], Y_indiff[:length], time_limit=10_800, 
                        n_threads=16, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4), relation_type="indifference")

        print(f"ObjVal for length {length}:", dist2.solver.ObjVal, "Optimization Status:", dist2.solver.Status)
        plt.figure(figsize=(12, 16))
        for i in range(4):
            plt.subplot(4, 2, i+1)
            plt.plot([dist2.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
            plt.plot([dist2.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
            plt.plot([dist2.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
            plt.plot([dist2.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        for i in range(4):
            plt.subplot(4, 2, i+5)
            plt.plot(generator.dms[0].coefficients[i], c="blue")
            plt.plot(generator.dms[1].coefficients[i], c="orange")
        plt.suptitle(dist2.solver.ObjVal)
        plt.savefig(f"xp_3/indiff_{length}_{dist2.solver.Status}_single.png")
        plt.show()

        coefficients = {
            "s1": [],
            "s2": [],
            "d1": [],
            "d2": [],
        }
        for name in coefficients.keys():
            for i in range(4):
                coefficients[name].append([dist2.marginal_coeffs[name, i, k].x for k in range(6)])
        np.save(f"xp_3/indiff_{length}_{dist2.solver.Status}_couple.npy", np.stack([list(coefficients.values())]))

        dist3 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3, lipschitz_coeff=1e-4)
        # dist.solver.setParam("DualReductions", 0)
        dist3.fit_generic(X_pref[:length], Y_pref[:length], time_limit=10_800,
                        n_threads=16, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4), relation_type="preference",
        clustering=np.concatenate([[i, i] for i in range(length//2)]))

        print(f"ObjVal for length {length}:", dist3.solver.ObjVal, "Optimization Status:", dist3.solver.Status)
        plt.figure(figsize=(12, 16))
        for i in range(4):
            plt.subplot(4, 2, i+1)
            plt.plot([dist3.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
            plt.plot([dist3.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
            plt.plot([dist3.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
            plt.plot([dist3.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        for i in range(4):
            plt.subplot(4, 2, i+5)
            plt.plot(generator.dms[0].coefficients[i], c="blue")
            plt.plot(generator.dms[1].coefficients[i], c="orange")
        plt.suptitle(dist3.solver.ObjVal)
        plt.savefig(f"xp_3/pref_{length}_{dist3.solver.Status}_couple.png")
        plt.show()

        coefficients = {
            "s1": [],
            "s2": [],
            "d1": [],
            "d2": [],
        }
        for name in coefficients.keys():
            for i in range(4):
                coefficients[name].append([dist3.marginal_coeffs[name, i, k].x for k in range(6)])
        np.save(f"xp_3/indiff_{length}_{dist3.solver.Status}_couple.npy", np.stack([list(coefficients.values())]))


        dist4 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3, lipschitz_coeff=1e-4)
        # dist.solver.setParam("DualReductions", 0)
        dist4.fit_generic(X_pref[:length], Y_pref[:length], time_limit=10_800,
                        n_threads=16, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4), relation_type="preference")

        print(f"ObjVal for length {length}:", dist4.solver.ObjVal, "Optimization Status:", dist4.solver.Status)
        plt.figure(figsize=(12, 16))
        for i in range(4):
            plt.subplot(4, 2, i+1)
            plt.plot([dist4.marginal_coeffs["s1", i, k].x for k in range(6)], c="blue", marker="x")
            plt.plot([dist4.marginal_coeffs["s2", i, k].x for k in range(6)], c="cyan")
            plt.plot([dist4.marginal_coeffs["d1", i, k].x for k in range(6)], c="red", marker="o")
            plt.plot([dist4.marginal_coeffs["d2", i, k].x for k in range(6)], c="orange")

        for i in range(4):
            plt.subplot(4, 2, i+5)
            plt.plot(generator.dms[0].coefficients[i], c="blue")
            plt.plot(generator.dms[1].coefficients[i], c="orange")
        plt.suptitle(dist4.solver.ObjVal)
        plt.savefig(f"xp_3/pref_{length}_{dist4.solver.Status}_single.png")
        plt.show()

        coefficients = {
            "s1": [],
            "s2": [],
            "d1": [],
            "d2": [],
        }
        for name in coefficients.keys():
            for i in range(4):
                coefficients[name].append([dist4.marginal_coeffs[name, i, k].x for k in range(6)])
        np.save(f"xp_3/indiff_{length}_{dist4.solver.Status}_couple.npy", np.stack([list(coefficients.values())]))
