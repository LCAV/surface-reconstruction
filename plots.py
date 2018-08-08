from matplotlib import pylab
import matplotlib.pyplot as plt
from signals import *

def plot_results(signal, color='b', resolution=100, interval=(0, 1), lw = 1, label="", ls='-'):
    t = np.linspace(interval[0], interval[1], resolution)
    return pylab.plot(t, signal.get_samples(t), color, linewidth=lw, label=label, ls=ls)


def stem_results(positions, samples, color='b',label=""):
    return pylab.stem(positions, samples,  markerfmt=color + 'o', linefmt=color + '-.',label=label)


def generate_plots(n, ovs, noise_scale, true_param, cut=0):
    version = str(n) + "_" + str(ovs) + "_" + str(noise_scale)

    errors = np.load('offline_results/offline_errors' + version + '.npy')
    betas = np.load('offline_results/offline_beta' + version + '.npy')
    parameters = np.load('offline_results/offline_params' + version + '.npy')

    errors = np.degrees(errors[cut:])
    betas = betas[cut:]
    parameters = np.degrees(parameters[cut:])

    fig1, ax0 = plt.subplots()
    ax0.semilogy(errors, 'r-')
    ax0.set_xlabel('iteration')
    ax0.set_ylabel('errors', color='r')
    ax0.tick_params('y', colors='r')

    ax1 = ax0.twinx()
    ax1.axhline(true_param)
    ax1.plot(parameters, 'c-')
    ax1.set_ylabel('parameters', color='c')
    ax1.tick_params('y', colors='c')

    fig1.tight_layout()

    fig2, ax2 = plt.subplots()
    ax2.semilogy(errors, 'r-')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('errors', color='r')
    ax2.tick_params('y', colors='r')

    ax3 = ax2.twinx()
    ax3.semilogy(np.abs(parameters - true_param), 'g-')
    ax3.set_ylabel('true error', color='g')
    ax3.tick_params('y', colors='g')

    fig2.tight_layout()
    return fig1, fig2


def known_error(start_positions,  model_size, tr_param, samples):
    x = SecondSurfacePolynomial.create_ls_matrix(start_positions, model_size, tr_param)
    parameter_estimate = np.linalg.solve(np.dot(x.T, x), np.dot(x.T, samples))
    return np.linalg.norm(np.dot(x, parameter_estimate) - samples)


def draw_plot(ax, data, edge_color, fill_color):
    bp = ax.boxplot(data, whis=[0, 95], patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def my_degrees(rad):
    return rad*180/np.pi