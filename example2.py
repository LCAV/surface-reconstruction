from samplers import *
from solvers import *
from plots import  *
from matplotlib import pylab
from pylab import rcParams
rcParams['figure.figsize'] = 5, 4

pylab.ion()
N = 10
start_param = [2,-1, 1]
polynomial = SignalPolynomial(start_param)
# print polynomial.degree
plot_results(polynomial,'b',lw =3)

# sampler = DeterministicSampler(polynomial,N)
sampler = GaussianSampler(polynomial,N,1,0.05)
# print sampler.get_positions()
# print sampler.get_samples()
# print sampler.number_samples
stem_results(sampler.get_positions(),sampler.get_samples(),'b')

# solver = OrdinaryLS(sampler.get_samples(),polynomial.degree,SignalPolynomial)
solver = AlternatingLS(sampler.get_samples(),polynomial.model_size,SignalPolynomial,False,stopping_error=1e-10,beta=0.01)
solver.solve()
# print solver.samples
# print solver.make_x()
# print solver.parameter_estimate
print(solver.train_error)
print(polynomial.square_error(SignalPolynomial(solver.parameter_estimate)))
pylab.ioff()
plot_results(SignalPolynomial(solver.parameter_estimate),'r',lw = 3)
stem_results(solver.get_position_estimates(),solver.get_samples(),'r');
pylab.show()
# stem_results(DeterministicSampler(SignalPolynomial(solver.parameter_estimate),N),'g')