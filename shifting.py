import itertools
from plots import *
from matplotlib import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4

"""Script generating plots describing the movement of samples for a path in the space of polynomials,
finaly replaced by the plots form unknown-locations.ipynb """

pol = SignalPolynomial([2, -1, 1])
pylab.show()
change = np.array([1, -1, 0])
alpha = np.linspace(0, 5)
sample_postitons = [0.2, 0.4, 0.6, 0.8]
colors = itertools.cycle(["b", "c", "g", "r"])


pylab.ion()
plot_results(pol, color='k')
for s in sample_postitons:
    stem_results([s], [pol.get_samples(s)], color=next(colors))
pylab.gcf().subplots_adjust(bottom=0.15)
# pylab.ylabel(r'$\alpha$',fontsize=20)
pylab.xlabel(r'$t$', fontsize=20)
pylab.xlim(0, 1)
pylab.ioff()
pylab.show()

pylab.ion()
for s in sample_postitons:
    print(pol.get_samples(s))
    p = pol.path(s, 0.1*change)
    pylab.plot(p, alpha, lw=3, color=next(colors))
pylab.gcf().subplots_adjust(bottom=0.15)
pylab.ylabel(r'$\alpha$', fontsize=20)
pylab.xlabel(r'$t$', fontsize=20)
pylab.xlim(0, 1)
pylab.ioff()
pylab.show()
