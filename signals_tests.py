"""
Test different signals and solvers combinations.
Gives example syntax.
"""


from samplers import *
from solvers import *

def test1(signal_type, sampler_type, solver_type, plot=False):
    n = 20
    start_param = [1, -2, 1]
    polynomial = signal_type(start_param)
    if plot:
        plot_results(polynomial, 'g')
    sampler = sampler_type(polynomial, n)
    if plot:
        stem_results(sampler.sample_positions, sampler.sample_values, 'g')
    solver = solver_type(sampler.sample_values, len(start_param), signal_type)
    solver.solve()
    if plot:
        plot_results(signal_type(solver.parameter_estimate))
        stem_results(solver.position_estimate, sampler.sample_values)
        pylab.show()
    print("{:.2e}".format(solver.train_error))
    print("{:.2e}".format(polynomial.square_error(solver.parameter_estimate)))


print('testing deterministic sampling + OLS:')
test1(SignalPolynomial, DeterministicSampler, OrdinaryLS)
print('testing deterministic sampling + ALS:')
test1(SignalPolynomial, DeterministicSampler, AlternatingLS)
print('testing deterministic sampling + "free" ALS:')
test1(SignalPolynomial, DeterministicSampler, AlternatingLS)

print('testing gaussian sapling + OLS:')
test1(SignalPolynomial, GaussianSampler, OrdinaryLS)
print('testing gaussian sapling + ALS:')
test1(SignalPolynomial, GaussianSampler, AlternatingLS, True)
