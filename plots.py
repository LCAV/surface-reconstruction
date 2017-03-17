from matplotlib import pylab
from signals import *

def plot_results(signal, color='b', resolution=100, interval=(0, 1), lw = 1, label="", ls='-'):
    t = np.linspace(interval[0], interval[1], resolution)
    return pylab.plot(t, signal.get_samples(t), color, linewidth=lw, label=label, ls=ls)


def stem_results(positions, samples, color='b',label=""):
    return pylab.stem(positions, samples,  markerfmt=color + 'o', linefmt=color + '-.',label=label)
