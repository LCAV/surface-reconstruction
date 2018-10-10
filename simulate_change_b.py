import numpy as np

from unwarping_functions import *

# --------------------------------------------------------#
# Settings
# --------------------------------------------------------#

K = 5
np.random.seed(100)
iter_max = 1000
noise_num = 50
SNR = np.linspace(-10, 40, noise_num)
atLeastOneSol = True
a_orig = np.random.uniform(-1, 1, 2*K+1)
a_orig = (a_orig + a_orig[::-1]) / 2.
redundancy = 400
x = np.linspace(-100, 100, 500000)
w = np.linspace(-np.pi, np.pi, 500000)
alpha_cl = dirichlet_inverse(1 / np.sin(2 * np.pi / (2 * K + 1)), K)
N = 30

b_list = np.array([0.6, 0.7, 1.0, 1.2, 2.3])
b_error_closest_b = np.zeros((len(b_list), noise_num))
h_error_closest_b = np.zeros((len(b_list), noise_num))
b_error_closest_h_n = np.zeros((len(b_list), noise_num))
h_error_closest_h_n = np.zeros((len(b_list), noise_num))
zero_sol_cases = np.zeros((len(b_list), noise_num))
multiple_sol_cases = np.zeros((len(b_list), noise_num))
alpha_list = np.zeros(len(b_list))


def compute_noise_iterator(noise_sig):
    counter = 0
    b_error_closest_b = 0
    h_error_closest_b = 0
    b_error_closest_h_n = 0
    h_error_closest_h_n = 0
    multiple_sol_cases = 0
    zero_sol_cases = 0

    for iter in range(iter_max):
        noise = np.random.normal(loc=0, scale=noise_sig, size=h_n.shape)
        h_n_noisy = h_n + noise
        unwarped = unwarp(h_n_noisy, K, T, sample_points, redundancy, atLeastOneSol)
        b_estimated = unwarped[0]
        a_estimated = unwarped[1]
        assert (len(b_estimated) == len(a_estimated))
        if len(b_estimated) == 0:
            zero_sol_cases += 1
        elif len(b_estimated) >= 1:
            b_error = np.array([np.abs(b_est - b_orig_periodized) / np.abs(b_orig_periodized) for b_est in b_estimated])
            h_n_hat = np.array(
                [compute_h_n_hat(a_est, b_est, K, sample_points) for a_est, b_est in zip(a_estimated, b_estimated)])
            h_error = np.array([np.linalg.norm(h_n - h_n_h, 2) / np.linalg.norm(h_n, 2) for h_n_h in h_n_hat])

            b_error_closest_b += b_error
            h_error_closest_b += h_error
            b_error_closest_h_n += b_error
            h_error_closest_h_n += h_error

            counter += 1
        if len(b_estimated) > 1:
            multiple_sol_cases += 1

    b_error_closest_b /= counter
    h_error_closest_b /= counter
    b_error_closest_h_n /= counter
    h_error_closest_h_n /= counter

    return b_error_closest_b, h_error_closest_b, b_error_closest_h_n, h_error_closest_h_n, multiple_sol_cases, zero_sol_cases


for b_ind, b_orig in enumerate(b_list):
    print(b_ind)
    alpha_orig = 2 * np.pi * b_orig / N
    alpha_list[b_ind] = alpha_orig
    T = (2 * K + 1.0) / N
    b_orig_periodized = (2 * K + 1) * np.abs(periodize_angle(alpha_orig)) / (2 * np.pi * T)

    n = np.arange(np.int(np.min(x) / T), np.int(np.max(x) / T) + 1)
    sample_points = n * T
    h_n = g_fun(b_orig * sample_points, a_orig, K)

    noise_sig_list = np.std(h_n) / (10**(SNR / 20.))

    p = Pool(cpu_count())
    results_pooled = p.map(compute_noise_iterator, noise_sig_list)
    p.close()
    for noise_ind in range(noise_num):
        b_error_closest_b[b_ind, noise_ind] = results_pooled[noise_ind][0]
        h_error_closest_b[b_ind, noise_ind]= results_pooled[noise_ind][1]
        b_error_closest_h_n[b_ind, noise_ind] = results_pooled[noise_ind][2]
        h_error_closest_h_n[b_ind, noise_ind] = results_pooled[noise_ind][3]
        multiple_sol_cases[b_ind, noise_ind] = results_pooled[noise_ind][4]
        zero_sol_cases[b_ind, noise_ind] = results_pooled[noise_ind][5]

np.savez('noise_vars_change_b_parallel.npz', zero_sol_cases=zero_sol_cases, multiple_sol_cases=multiple_sol_cases,
         iter_max=iter_max, SNR=SNR, b_error_closest_b=b_error_closest_b, h_error_closest_b=h_error_closest_b,
         b_error_closest_h_n=b_error_closest_h_n, h_error_closest_h_n=h_error_closest_h_n, a_orig=a_orig,
         b_list=b_list, alpha_list=alpha_list)
