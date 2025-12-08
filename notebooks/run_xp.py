import sys

print(sys.executable)
print("ole")
print(sys.version)
import matplotlib.pyplot as plt
import gurobipy as hp
sys.path.append("../")


import pickle
import numpy as np

from python.data_generation import SyntheticDataGenerator
from python.distances import TwoUTASpaceDiameter

alldist = {}
for ndata in [10, 20, 100, 1_000]:
    print("Nb of data:", ndata)
    generator = SyntheticDataGenerator(
    	n_dms=2,
    	n_criteria=4,
    	method_params={"n_pieces": 5},
	)

    X, Y, info = generator.generate_preferences(num_pairs=ndata, return_clusters=True)
    dist = TwoUTASpaceDiameter(n_pieces=5)
    dist.fit(X, Y)

    alldist[ndata] = {
        "s1": [[dist.marginal_coeffs["s1", i, k].x for k in range(6)] for i in range(4)],
        "s2": [[dist.marginal_coeffs["s2", i, k].x for k in range(6)] for i in range(4)],
        "d1": [[dist.marginal_coeffs["d1", i, k].x for k in range(6)] for i in range(4)],
        "d2": [[dist.marginal_coeffs["d2", i, k].x for k in range(6)] for i in range(4)],
        "objval": dist.solver.objVal
    }

    with open('filename_bis.pickle', 'wb') as handle:
        pickle.dump(alldist, handle, protocol=pickle.HIGHEST_PROTOCOL)
