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
    X, Y, info = generator.generate_indifferences_alldms(num_pairs=200, return_clusters=True, return_utilities=True)

    for length in [100, 200]:

        dist = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3)
        # dist.solver.setParam("DualReductions", 0)
        dist.fit_from_coupled_indifferences(X[:length], Y[:length], time_limit=10_800, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4), warm_zs=True)

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
        plt.savefig(f"myres_{length}_{dist.solver.Status}_couple.png")
        plt.show()


        dist2 = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-4, lipschitz_coeff=2e-4)
        # dist.solver.setParam("DualReductions", 0)
        dist2.fit_from_indifferences(X[:length], Y[:length], time_limit=10_800, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4))

        print(f"ObjVal for length {length}:", dist.solver.ObjVal, "Optimization Status:", dist.solver.Status)
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
        plt.suptitle(dist.solver.ObjVal)
        plt.savefig(f"myres_{length}_{dist.solver.Status}_single.png")
        plt.show()

    X, Y, info = generator.generate_indifferencgees_alldms(num_pairs=200, return_clusters=True, return_utilities=True)

    for length in [100, 200]:

        dist = TwoUTASpaceDiameter(n_pieces=5, epsilon=1e-3)
        # dist.solver.setParam("DualReductions", 0)
        dist.fit_from_coupled_indifferences(X[:length], Y[:length], time_limit=10_800, inflexions=np.vstack([np.linspace(0, 1., 6)] * 4), warm_zs=True)

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
        plt.savefig(f"myres_{length}_{dist.solver.Status}_couple.png")
        plt.show()
