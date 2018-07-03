import pickle
import numpy.random as nr
import numpy as np
from sortedcontainers import SortedSet

test_set = SortedSet([])
all_degrees = SortedSet([])
new_params = False
n_tests = 100

noises = np.linspace(-6, 1, 100)

for polynomial_degree in [5]:
    for ovs in [1, 2, 4, 8]:
        for noise_scale in noises[50:]:
            test_set.add((polynomial_degree, ovs, noise_scale))
            all_degrees.add(polynomial_degree)

if new_params:
    for polynomial_degree in all_degrees:
        params = 2 * nr.randn(n_tests, polynomial_degree)
        params[:, 0] = 1
        np.savetxt("polynomials" + str(polynomial_degree) + ".csv", params, delimiter=",")

with open("test_set", "wb") as out_file:
    pickle.dump(list(test_set), out_file)