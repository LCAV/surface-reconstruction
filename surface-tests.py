from samplers import *
from solvers import *
from plots import *

# set parameters
tests = 50  # number of tests
n = 4       # number of parameters of the polynomial (degree + 1)
f = 1.0     # distance between the origin and the image plane
b = 1.0     # intersection between camera axis and the surface
slopes = np.linspace(-np.pi/6, np.pi/6, 13)

# containers for results:
errors = []
nsr = []

for test in range(tests):
    print("test: ", test)

    tmp_err = []
    start_param = nr.randn(n, 1)

    print(start_param)

    for slope in slopes:
        polynomial = SecondSurfacePolynomial(start_param)
        sampler = SurfaceSampler(polynomial, 2*n, [slope, b, f], interval_length=2, sigma=0.0, beg=-1)
        noise = 1e-1*nr.randn(2*n)
        nsr.append(np.linalg.norm(noise) / np.linalg.norm(sampler.sample_values))
        sample_values = sampler.sample_values + noise

        solver = ConstrainedALS(
            sample_values,
            polynomial.model_size,
            SecondSurfacePolynomial,
            start_pos=sampler.sample_positions,
            stopping_error=1e-10,
            beta=0.1,
            show_plots=False,
            max_iter=10000,
            fl=f)

        true_error = 0
        try:
            solver.solve()
            true_error = abs(slope - solver.tr_param[0])
        except AssertionError as as_err:
            print("assertion error:", as_err.args[0])
            true_error = np.NAN

        tmp_err.append(true_error)
    errors.append(tmp_err)

errors = np.array(errors)
nsr = np.array(nsr)

print("mean:", np.nanmean(errors)*180/np.pi)
print("median:", np.nanmedian(errors))
print("std:", np.nanstd(errors))
print("NANS:", str(np.count_nonzero(np.isnan(errors))/len(errors.flatten())*100)+"%")
print("noise to signal", np.nanmean(nsr))
