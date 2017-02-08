from plots import *
import numpy as np
from matplotlib import pylab


def cost(alpha, positions, samples, model_size, signal_type):
    x = signal_type.create_ls_matrix(positions, model_size, alpha)
    return np.linalg.norm(
        samples - np.dot(
            x,
            np.linalg.solve(
                np.dot(x.T, x),
                np.dot(x.T, samples)
            )
        )
    )


def one_tr_parameter(polyn_params, alpha_range, log=False):
    signal = SurfacePolynomial(polyn_params)
    pylab.ion()

    for n in range(len(polyn_params)+1, 2 * len(polyn_params)+2):
        positions = np.linspace(0.01, 0.9, n)
        samples = signal.get_samples(positions)
        c = [cost(a, positions, samples, len(polyn_params), SurfacePolynomial) for a in alpha_range]
        if log:
            pylab.semilogy(alpha_range, c)
        else:
            pylab.plot(alpha_range, c)
    pylab.ioff()
    pylab.show()


def two_tr_parameters(polyn_params, positions):
    n = 200
    signal = FullSurfacePolynomial(polyn_params)
    samples = signal.get_samples(positions)
    alpha = np.linspace(-1, 0.4, n)
    beta = np.linspace(0.4, 2, n)
    a, b = np.meshgrid(alpha, beta)
    c = np.array([cost(p, positions, samples, len(polyn_params), FullSurfacePolynomial)
                  for p in zip(np.ravel(a), np.ravel(b))])
    c = c.reshape(a.shape)
    pylab.pcolormesh(a, b, c)
    pylab.show()


parameters = [1, 1, 1]
one_tr_parameter(parameters, np.linspace(-1, 1, 500), log=True)
# two_tr_parameters(parameters, np.linspace(0.01, 0.9, 2 * len(parameters)))
