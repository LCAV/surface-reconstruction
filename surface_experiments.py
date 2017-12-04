from samplers import *
from solvers import *
from plots import *
from multiprocessing import Pool
import matplotlib.pyplot as plt

def test_block(beginning):
    errors_ = []
    nsr_ = []
    np.random.seed(beginning)
    for test in range(int(tests / pools)):
        test_number = int(tests / pools) * beginning + test
        print("test: ", test_number)

        tmp_err = []
        start_param = params[test_number,:]

        for slope in slopes:
            polynomial = SecondSurfacePolynomial(start_param)
            sampler = SurfaceSampler(polynomial, 2 * ovs * n, [slope, b, f], interval_length=2, sigma=0.0, beg=-1)
            noise = noise_ampl * nr.randn(2 * ovs * n)
            nsr_.append(np.linalg.norm(noise) / np.linalg.norm(sampler.sample_values))
            sample_values = sampler.sample_values + noise

            solver = ConstrainedALS(
                sample_values,
                polynomial.model_size,
                SecondSurfacePolynomial,
                start_pos=sampler.sample_positions,
                stopping_error=1e-16,
                beta=0.1/ovs,
                show_plots=plots,
                max_iter=1000,
                fl=f,
                verbose=verbose)

            try:
                solver.solve()
                true_error = abs(slope - solver.tr_param[0])
            except AssertionError as as_err:
                print("assertion error:", as_err.args[0])
                true_error = np.NAN

            if(verbose):
                np.save("offline_errors"+version, solver.error_over_time)
                np.save("offline_params" + version, solver.tr_params_over_time)
                np.save("offline_beta" + version, solver.beta_over_time)

            tmp_err.append(true_error)
        errors_.append(np.array(tmp_err))
    return errors_, nsr_,


# set parameters
save = True
new_params = False
plots = False
verbose = True
tests = 100  # number of tests (should be at least two, because)
pools = 1
directory = "results/"


for n in range(7,10):

    ovs = 1  # oversampling
    f = 1.0  # distance between the origin and the image plane
    b = 1.0  # intersection between camera axis and the surface
    slopes = np.linspace(-np.pi / 9, np.pi / 9, 13)
    print("Starting slope:", np.degrees(slopes[0]))

    # create or load the polynomials
    if new_params:
        params = 2 * nr.randn(tests, n)
        params[:, 0] = 1
        np.savetxt("polynomials"+str(n)+".csv", params, delimiter=",")

    params = np.loadtxt("polynomials"+str(n)+".csv", delimiter=",")

    if plots:
        for i in range(tests):
            t = np.linspace(-1, 1, 100)
            pol = SignalPolynomial(params[i, :])
            plt.plot(t, pol.get_samples(t))
        plt.show()

    for ovs in [1,2,4,8]:
        for noise_scale in range(1,5):
            noise_ampl = 0.0
            if noise_scale is not 0:
                noise_ampl = 10.0 ** (-noise_scale)

            version = str(n) + "_" + str(ovs) + "_" + str(noise_scale)
            print("starting experiments for version: " + version)

            errors, nsr = test_block(0)
            errors = np.array(errors)
            nsr = np.array(nsr)

            print("mean:", np.degrees(np.nanmean(errors)))
            print("median:", np.degrees(np.nanmedian(errors)))
            print("std:", np.degrees(np.nanstd(errors)))
            print("NANS:", str(np.count_nonzero(np.isnan(errors)) / len(errors.flatten()) * 100) + "%")
            print("noise to signal", np.nanmean(nsr))

            if save:
                np.save(directory+"errors_"+version, errors)
                np.save(directory+"nsr_"+version, nsr)
                np.save(directory+"params_"+version, params)
