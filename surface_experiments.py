from samplers import *
from solvers import *
from plots import *
from multiprocessing import Pool
import pickle

def test_block(n, ovs, nl, tests, slopes, verbose, save, directory, b, f):

    errors_ = []
    sig_pow = []
    np.random.seed(10*n + ovs)
    noise_ampl = 0.0
    if nl is not 0:
        noise_ampl = 10.0 ** (-nl)

    version = str(n) + "_" + str(ovs) + "_" + "{0:.2f}".format(nl)
    print("starting version:", version)

    params = np.loadtxt("polynomials"+str(n)+".csv", delimiter=",")
    if tests > params.shape[0]:
        print("not enough polynomials!")
        return

    for test in range(tests):
        tmp_err = []
        start_param = params[test,:]

        for slope in slopes:
            polynomial = SecondSurfacePolynomial(start_param)
            sampler = SurfaceSampler(polynomial, 2 * ovs * n, [slope, b, f], interval_length=2, sigma=0.0, beg=-1)
            noise = noise_ampl * nr.randn(2 * ovs * n)
            sig_pow.append(np.mean(np.power(sampler.sample_values,2)))
            sample_values = sampler.sample_values + noise

            solver = ConstrainedALS(
                sample_values,
                polynomial.model_size,
                SecondSurfacePolynomial,
                start_pos=sampler.sample_positions,
                stopping_error=1e-14,
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
            except Exception as other_exc:
                print("unexpected error:", other_exc.args[0])
                true_error = np.NAN

            if(verbose):
                np.save("offline_errors" + version, solver.error_over_time)
                np.save("offline_params" + version, solver.tr_params_over_time)
                np.save("offline_beta" + version, solver.beta_over_time)

            tmp_err.append(true_error)
        errors_.append(np.array(tmp_err))
    errors_ = np.array(errors_)
    sig_pow = np.array(sig_pow)

    print("finished version:", version)
    if verbose:
        print("mean:", np.degrees(np.nanmean(errors_)))
        print("median:", np.degrees(np.nanmedian(errors_)))
        print("std:", np.degrees(np.nanstd(errors_)))
        print("NANS:", str(np.count_nonzero(np.isnan(errors_)) / len(errors_.flatten()) * 100) + "%")
        print("signal_power:", np.mean(sig_pow))

    if save:
        np.save(directory + "errors_" + version, errors_)
        np.save(directory + "pow_" + version, sig_pow)
        np.save(directory + "params_" + version, params)
        print("saved version:", version)
    return errors_, sig_pow,


if __name__ == '__main__':

    # set parameters
    save = False
    plots = False
    verbose = False
    n_tests = 100  # number of tests (should be at least two, because)
    directory = "results/"
    f = 1.0  # distance between the origin and the image plane
    b = 1.0  # intersection between camera axis and the surface
    slopes = np.linspace(-np.pi / 9, np.pi / 9, 13)


    def test_block_unpack(t):
        return test_block(t[0], t[1], t[2], n_tests, slopes, verbose, save, directory, b, f)


    with open('test_set', 'rb') as in_file:
        test_set = pickle.load(in_file)

        for t in test_set:
            print(t)

        # start 4 worker processes
        pool = Pool(processes=4)
        pool.map(test_block_unpack, test_set)