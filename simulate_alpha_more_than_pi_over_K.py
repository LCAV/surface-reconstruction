import numpy as np

from unwarping_functions import *

# --------------------------------------------------------#
# Settings
# --------------------------------------------------------#

K = 5
np.random.seed(100)
iter_max = 10000
noise_num = 100
SNR = np.linspace(-20, 100, noise_num)
atLeastOneSol = True
a_orig = np.random.uniform(-1, 1, 2*K+1)
a_orig = (a_orig + a_orig[::-1]) / 2.
redundancy = 300
x = np.linspace(-100, 100, 1000000)
w = np.linspace(-np.pi, np.pi, 1000000)
b_orig = 4
T = (2 * K + 1) / (2 * np.pi * b_orig) * dirichlet_inverse((2 * (2 * K + 1)) / (3 * np.pi), K)
T = T * 2.3
alpha_orig = 2 * np.pi / (2 * K + 1) * T * b_orig
b_orig_periodized = (2 * K + 1) * np.abs(periodize_angle(alpha_orig)) / (2 * np.pi * T)

n = np.arange(np.int(np.min(x) / T), np.int(np.max(x) / T) + 1)
sample_points = n * T
h_n = g_fun(b_orig * sample_points, a_orig, K)

# --------------------------------------------------------#
# Noise study
# --------------------------------------------------------#

noise_sig_list = np.std(h_n) / (10**(SNR / 20.))
b_error_closest_b = np.zeros(len(noise_sig_list))
h_error_closest_b = np.zeros(len(noise_sig_list))
b_error_closest_h_n = np.zeros(len(noise_sig_list))
h_error_closest_h_n = np.zeros(len(noise_sig_list))
zero_sol_cases = np.zeros(len(noise_sig_list))
multiple_sol_cases = np.zeros(len(noise_sig_list))

for noise_ind, noise_sig in enumerate(noise_sig_list):
    print('computing noise number ' + str(noise_ind+1) + ' out of ' + str(noise_num))
    counter = 0
    for iter in range(iter_max):
        noise = np.random.normal(loc=0, scale=noise_sig, size=h_n.shape)
        h_n_noisy = h_n + noise
        unwarped = unwarp(h_n_noisy, K, T, sample_points, redundancy, atLeastOneSol)
        b_estimated = unwarped[0]
        a_estimated = unwarped[1]
        assert(len(b_estimated) == len(a_estimated))
        if len(b_estimated) == 0:
            zero_sol_cases[noise_ind] += 1
        elif len(b_estimated) >= 1:
            b_error = np.array([np.abs(b_est - b_orig_periodized) / np.abs(b_orig_periodized) for b_est in b_estimated])
            h_n_hat = np.array([compute_h_n_hat(a_est, b_est, K, sample_points) for a_est, b_est in zip(a_estimated, b_estimated)])
            h_error = np.array([np.linalg.norm(h_n - h_n_h, 2) / np.linalg.norm(h_n, 2) for h_n_h in h_n_hat])

            b_error_closest_b[noise_ind]   += b_error[np.argmin(b_error)]
            h_error_closest_b[noise_ind]   += h_error[np.argmin(b_error)]
            b_error_closest_h_n[noise_ind] += b_error[np.argmin(h_error)]
            h_error_closest_h_n[noise_ind] += h_error[np.argmin(h_error)]

            counter += 1
        if len(b_estimated) > 1:
            multiple_sol_cases[noise_ind] += 1

    b_error_closest_b[noise_ind] /= counter
    h_error_closest_b[noise_ind] /= counter
    b_error_closest_h_n[noise_ind] /= counter
    h_error_closest_h_n[noise_ind] /= counter

np.savez('noise_vars_non-unique_alpha_moreThanPiK.npz', zero_sol_cases=zero_sol_cases, multiple_sol_cases=multiple_sol_cases,
         iter_max=iter_max, h_n=h_n, SNR=SNR, b_error_closest_b=b_error_closest_b, h_error_closest_b=h_error_closest_b,
         b_error_closest_h_n=b_error_closest_h_n, h_error_closest_h_n=h_error_closest_h_n, a_orig=a_orig, b_orig=b_orig,
         T=T)
