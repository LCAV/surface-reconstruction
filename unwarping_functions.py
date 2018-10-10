import numpy as np
import math

def periodize_angle(theta):
    result = np.array(theta % (2.0 * np.pi))
    idx = result > np.pi
    result[idx] -= 2 * np.pi
    return result


def dirichlet(w, K):
    """Drichlet kernel
    :param w: The argument of the kernel.
    :param K: order of the Dirichlet kernel.
    :return: The values of the Dirichlet kernel."""

    return np.sin(w * (K+0.5)) / np.sin(w/2.)


def dirichlet_inverse(y, K, threshold=0.01):
    """Numerial inverse of the Drichlet kernel
    :param y: Value of the kernel.
    :param K: order of the Dirichlet kernel.
    :param threshold: numerical accuracy of the numerical inverse.
    :return: The values of the Dirichlet kernel."""
    w_solve = np.linspace(0, np.pi+np.pi/10000, 10000)
    y_tmp = np.abs(dirichlet(w_solve, K) - y)
    idx = np.argwhere(np.isclose(y_tmp[:-1], np.zeros(y_tmp[:-1].shape), atol=threshold)).reshape(-1)
    min_inds = np.array([], dtype=int)
    for i in idx:
        if y_tmp[i-1] > y_tmp[i] and y_tmp[i] < y_tmp[i+1]:
            min_inds = np.hstack((min_inds, i))
    result = w_solve[min_inds]
    return result


def g_fun(x, a, K):
    g_x = np.zeros(len(x)) + 1j * np.zeros(len(x))
    k = np.arange(-K, K+1)
    for k in range(2*K+1):
        g_x = g_x + a[k] * np.exp(1j * 2 * np.pi * (k - K) * x / (2 * K + 1))
    return g_x


def estimate_theta(h_n, L, redundancy):
    myL = L + 30
    N = 2 * myL + 1 + redundancy
    D = np.zeros((N - myL, myL))
    for i in range(1, myL + 1):
        D[:, (i-1)] = h_n[np.arange((myL-i),(N-i))]

    d = h_n[np.arange(myL, N)]
    aa = np.linalg.lstsq(D, d)[0]
    mu = np.roots(np.hstack((1, -aa)))
    idx = np.argsort(np.abs(np.abs(mu) - 1))[:L]
    mu = mu[idx]
    theta_estimated = np.sort(np.angle(mu))
    return theta_estimated


def find_s(theta):
    s = np.real(np.sum(np.exp(1j * theta)))
    return s


def estimate_alpha(s, K, version='new'):
    if version == 'new':
        def myFun(w_solve):
            return dirichlet(w_solve, K) - s
        alpha_hat = roots(myFun, 0, np.pi, eps=1e-4)[1:]
    elif version == 'old':
        w_solve = np.linspace(0, np.pi + np.pi/100000000, 100000000)
        g1 = np.abs(dirichlet(w_solve, K) - s)
        idx = np.argwhere(np.isclose(g1[:-1], np.zeros(g1[:-1].shape), atol=1e-02)).reshape(-1)
        min_inds = np.array([], dtype=int)
        for i in idx:
            if g1[i-1] > g1[i] and g1[i] < g1[i+1]:
                min_inds = np.hstack((min_inds, i))
        alpha_hat = w_solve[min_inds]
    return alpha_hat


def filter_alpha(alpha_hat, theta_estimated, K, threshold=0.1, atLeastOneSol=False):
    if len(alpha_hat) == 0:
        return alpha_hat
    else:
        theta_test = [None] * len(alpha_hat)
        for i in range(len(alpha_hat)):
            theta_test[i] = np.sort(periodize_angle(np.arange(-K, K + 1) * alpha_hat[i]))
        errors = np.array([np.linalg.norm(theta_estimated - np.sort(t_test), 2) for t_test in theta_test])
        ind1 = np.argmin(errors)
        ind2 = errors <= threshold
        if np.sum(ind2) == 0 and atLeastOneSol:
            return np.array([alpha_hat[ind1]]).flatten()
        else:
            return np.array([alpha_hat[ind2]]).flatten()


def estimate_a(b_hat, h_n, K, sample_points):
    sampling_matrix = np.zeros((len(sample_points), 2*K+1)) + 1j * np.zeros((len(sample_points), 2*K+1))
    for k in range(2*K+1):
        sampling_matrix[:, k] = np.exp(1j * 2 * np.pi * (k-K) * b_hat * sample_points / (2*K+1))
    a_hat = np.dot(np.linalg.pinv(sampling_matrix), h_n)
    return np.real(a_hat)


def estimate_b(alpha_hat, K, T):
    b_hat = (2 * K + 1) * alpha_hat / (2 * np.pi * T)
    return b_hat


def unwarp(h_n, K, T, sample_points, redundancy, atLeastOneSol=True):
    theta_estimated = estimate_theta(h_n, 2*K+1, redundancy)
    s = find_s(theta_estimated)
    alpha_estimated = estimate_alpha(s, K)

    alpha_estimated_filtered = filter_alpha(alpha_hat=alpha_estimated, theta_estimated=theta_estimated, K=K,
                                            atLeastOneSol=atLeastOneSol)
    b_estimated = estimate_b(alpha_estimated_filtered, K, T)
    if np.size(b_estimated) >= 1:
        a_estimated = np.array([estimate_a(b_est, h_n, K, sample_points) for b_est in b_estimated])
    else:
        a_estimated = np.array([])
    return [b_estimated, a_estimated]


def compute_h_n_hat(a_hat, b_hat, K, sample_points):
    assert(len(a_hat) == (2*K+1))
    result = np.zeros(sample_points.shape) + 1j * np.zeros(sample_points.shape)
    for k in range(2*K+1):
        result += a_hat[k] * np.exp(1j * 2 * np.pi * (k-K) * b_hat * sample_points / (2*K+1))
    return np.real(result)


# taken from here: https://stackoverflow.com/questions/13054758/python-finding-multiple-roots-of-nonlinear-equation
def rootsearch(f, a, b, dx):
    x1 = a
    f1 = f(a)
    x2 = a + dx
    f2 = f(x2)
    while f1 * f2 > 0.0:
        if x1 >= b:
            return None,None
        x1 = x2
        f1 = f2
        x2 = x1 + dx
        f2 = f(x2)
    return x1, x2


def bisect(f, x1, x2, switch=0, epsilon=1.0e-9):
    f1 = f(x1)
    if f1 == 0.0:
        return x1
    f2 = f(x2)
    if f2 == 0.0:
        return x2
    if f1 * f2 > 0.0:
        print('Root is not bracketed')
        return None
    n = int(math.ceil(math.log(abs(x2 - x1)/epsilon)/math.log(2.0)))
    for i in range(n):
        x3 = 0.5 * (x1 + x2)
        f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if f2 * f3 < 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
    return (x1 + x2) / 2.0


def roots(f, a, b, eps=1e-6):
    result = []
    while 1:
        x1, x2 = rootsearch(f, a, b, eps)
        if x1 is not None:
            a = x2
            root = bisect(f,x1,x2,1)
            if root is not None:
                pass
                result.append(root)
        else:
            break
    return np.array(result)
