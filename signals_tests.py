from samplers import *
from solvers import *


def test1(signalType, samplerType, solverType, plot=False):
    n = 20
    start_param = [1, -2, 1]
    polynomial = signalType(start_param)
    if plot:
        plot_results(polynomial, 'g')
    sampler = samplerType(polynomial, n)
    if plot:
        stem_results(sampler.sample_positions, sampler.sample_values, 'g')
    solver = solverType(sampler.sample_values, len(start_param), signalType)
    solver.solve()
    if plot:
        plot_results(signalType(solver.parameter_estimate))
        stem_results(solver.position_estimate, sampler.sample_values)
        pylab.show()
    print("{:.2e}".format(solver.train_error))
    print("{:.2e}".format(polynomial.square_error(signalType(solver.parameter_estimate))))

print('testing deterministic sampling + OLS:')
test1(SignalPolynomial, DeterministicSampler, OrdinaryLS)
print('testing deterministic sampling + ALS:')
test1(SignalPolynomial, DeterministicSampler, AlternatingLS)
# print 'testing deterministic sampling + "free" ALS:'
# test1(SignalPolynomial, DeterministicSampler, AlternatingLS)

print('testing gaussian sapling + OLS:')
test1(SignalPolynomial, GaussianSampler, OrdinaryLS)
print('testing gaussian sapling + ALS:')
test1(SignalPolynomial, GaussianSampler, AlternatingLS,True)
# print 'testing "free" gaussian sampling + "free" ALS'
# test1(SignalPolynomial, GaussianSamplerFree, AlternatingLS,True)



# print 'testing deterministic sampling + ILS:'
# test1(SignalPolynomial, DeterministicSampler, InvertedLS)
# print 'testing gaussian sampling + ILS'
# test1(SignalPolynomial, GaussianSampler, InvertedLS)
