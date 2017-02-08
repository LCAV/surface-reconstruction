from samplers import *
from solvers import *
from plots import  *
from matplotlib import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4

pylab.ion()
N = 6
start_param = [2,-1, 1]
polynomial = FullSurfacePolynomial(start_param)
plot_results(polynomial,'g',lw =3)

sampler = DeterministicSampler(polynomial,N)
stem_results(sampler.get_positions(),sampler.get_samples(),'g',label="original")

new_pos = FullSurfacePolynomial.shifted_positions(sampler.get_positions(),trace_param=[-0.5,-0.5])
stem_results(new_pos,sampler.get_samples(),'r',label="start. positions")

solver = ConstrainedALS(
    sampler.get_samples(),
    polynomial.model_size,
    SurfacePolynomial,
    start_pos=new_pos,
    stopping_error=1e-16,
    beta=0.1,
    show_plots = False
    )
solver.solve()

# print solver.train_error
# print polynomial.square_error(SignalPolynomial(solver.parameter_estimate))
pylab.ioff()
# plot_results(SignalPolynomial(solver.parameter_estimate),'r',lw = 3)
stem_results(solver.get_position_estimates(),solver.get_samples(),'b',label="positions found")
pylab.xlabel(r'$t$',fontsize=20)
pylab.legend(loc="lower right")
pylab.gcf().subplots_adjust(bottom=0.15)
pylab.show()

