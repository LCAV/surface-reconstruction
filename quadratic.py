from plots import *
from matplotlib import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4

pylab.ion()
N = 10
start_param = np.array([2, -1, 1])
polynomial = SignalPolynomial(start_param)
plot_results(polynomial, 'k', lw=1)
r = np.roots(start_param+np.array([0, 0, -1.5]))
stem_results([r[0]], [1.5], 'r')

r = np.roots(start_param+np.array([0, 0, -1.2]))
stem_results([r[0]], [1.2], 'g')

r = np.roots(start_param+np.array([0, 0, -0.9]))
stem_results([r[0]], [0.9], 'c')
stem_results([r[1]], [0.9], 'b')

change = np.array([0, -1, 1])
change2 = np.array([1, -1, 0])

# for i in np.linspace(0.5,2,4):
#     polynomial=SignalPolynomial(start_param+i*change)
#     plot_results(polynomial,'r')
#     r = np.roots(start_param2+i*change2+np.array([0,0,-1.1]))
#     stem_results([r[0]],[1.1],'r')
pylab.ioff()
pylab.show()
