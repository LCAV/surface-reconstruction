""" Prepare Tests

Script generating and a set of parameters for simulations.
Parameters are saved as set in `parameters/test_set`

To use just run

    python test_set

script does not take any command line arguments or flags.

The script is intended to provide a simple way to describe
what experiments to perform.

"""

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
    for oversampling in [1, 2, 4, 8]:
        for noise_scale in noises[50:]:
            test_set.add((polynomial_degree, oversampling, noise_scale))
            all_degrees.add(polynomial_degree)

if new_params:
    for polynomial_degree in all_degrees:
        params = 2 * nr.randn(n_tests, polynomial_degree)
        params[:, 0] = 1
        np.savetxt("parameters/polynomials{}.csv.".format(polynomial_degree), params, delimiter=",")

with open("parameters/test_set", "wb") as out_file:
    pickle.dump(list(test_set), out_file)
