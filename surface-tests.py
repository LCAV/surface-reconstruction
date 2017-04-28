from samplers import *
from solvers import *
from plots import *
from multiprocessing import Pool

# set parameters
tests = 50  # number of tests
pools = 5
n = 6  # number of parameters of the polynomial (degree + 1)
ovs = 1  # oversampling
f = 1.0  # distance between the origin and the image plane
b = 1.0  # intersection between camera axis and the surface
slopes = np.linspace(-np.pi / 9, np.pi / 9, 13)
# slopes = slopes[1:-1]
noise_scale = 0
noise_ampl = 0.0
if noise_scale is not 0:
    noise_ampl = 10.0 ** (-noise_scale)


# containers for results:

def test_block(beginning):
    errors = []
    nsr = []
    params = []
    np.random.seed(beginning)
    for test in range(int(tests / pools)):
        print("test: ", int(tests / pools) * beginning + test)

        tmp_err = []
        start_param = nr.randn(n, 1)
        start_param[0] = 1

        # print(start_param)
        params.append(start_param)

        for slope in slopes:
            polynomial = SecondSurfacePolynomial(start_param)
            sampler = SurfaceSampler(polynomial, 2 * ovs * n, [slope, b, f], interval_length=2, sigma=0.0, beg=0)
            noise = noise_ampl * nr.randn(2 * ovs * n)
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
                fl=f,
                verbose=False)

            true_error = 0
            try:
                solver.solve()
                true_error = abs(slope - solver.tr_param[0])
            except AssertionError as as_err:
                print("assertion error:", as_err.args[0])
                true_error = np.NAN

            tmp_err.append(true_error)
        errors.append(np.array(tmp_err))
    return (errors, nsr, params)


p = Pool(pools)
sth = p.map(test_block, range(pools))
sth = np.array(sth)
errors = np.concatenate(sth[:, 0])
nsr = np.concatenate(sth[:, 1]).reshape(tests, len(slopes))
params = np.concatenate(sth[:, 2]).reshape(tests, n)

print("mean:", np.degrees(np.nanmean(errors)))
print("median:", np.degrees(np.nanmedian(errors)))
print("std:", np.degrees(np.nanstd(errors)))
print("NANS:", str(np.count_nonzero(np.isnan(errors)) / len(errors.flatten()) * 100) + "%")
print("noise to signal", np.nanmean(nsr))

version = str(n)+"_"+str(ovs)+"_"+str(noise_scale)
np.save("errors_"+version, errors)
np.save("nsr_"+version, nsr)
np.save("params_"+version, params)
