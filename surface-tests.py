from samplers import *
from solvers import *
from plots import  *

tests = 100
n = 4
f = 1.0
b = 10.0
errors = []
polynomials = []
slopes = []

for test in range(tests):
    print("test: ", test)
    start_param =0.01*nr.randn(n, 1)
    polynomials.append(start_param)
    # print(start_param)

    polynomial = SecondSurfacePolynomial(start_param)
    slope = 0.5*nr.rand()-0.5
    slopes.append(slope)

    sampler = SurfaceSampler(polynomial,2*n,[slope,b,f], interval_length=1, sigma=0.0)
    # noise = 1e-2*nr.randn(N)
    sample_values = sampler.get_samples()

    solver = ConstrainedALS(
        sample_values,
        polynomial.model_size,
        SecondSurfacePolynomial,
        start_pos=sampler.get_positions(),
        stopping_error=1e-10,
        beta=0.5,
        show_plots=False,
        max_iter=10000,
        fl=f)

    try:
        solver.solve()
    except AssertionError as as_err:
        print("assertion error:", as_err.args[0])
        solver.error = np.NAN

    errors.append(solver.error)

print("mean:", np.nanmean(errors))
print("std:",np.nanstd(errors))
print("NANS:", str(np.count_nonzero(np.isnan(errors))/tests*100)+"%")
