from samplers import *
from solvers import *
from plots import  *

tests = 3
n = 4
f = 1.0
b = 1.0
errors = []
nsr = []
slopes = np.linspace(-np.pi/6,np.pi/6,13)

for test in range(tests):
    print("test: ", test)

    tmp_err = []
    start_param = nr.randn(n, 1)
    # polynomials.append(start_param)
    print(start_param)

    for slope in slopes:
        true_error = 0
        polynomial = SecondSurfacePolynomial(start_param)
        sampler = SurfaceSampler(polynomial,2*n,[slope,b,f], interval_length=2, sigma=0.0, beg=-1)

        noise = 1e-1*nr.randn(2*n)
        sample_values = sampler.sample_values + noise
        nsr.append(np.linalg.norm(noise)/np.linalg.norm(sampler.sample_values))

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

        try:
            solver.solve()
            true_error = abs(slope - solver.tr_param[0])
        except AssertionError as as_err:
            print("assertion error:", as_err.args[0])
            true_error = np.NAN

        tmp_err.append(true_error)
        # print("slope errpr", true_error)
    errors.append(tmp_err)

errors = np.array(errors)
nsr = np.array(nsr)
print("mean:", np.nanmean(errors)*180/np.pi)
print("median:", np.nanmedian(errors))
print("std:",np.nanstd(errors))
print("NANS:", str(np.count_nonzero(np.isnan(errors))/len(errors.flatten())*100)+"%")
print("noise to signal", np.nanmean(nsr))
