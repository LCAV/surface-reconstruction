from solvers import *
from plots import *

# Example how to use Solver class
s = Solver(samples=[0, 1, 2], model_size=2, model_type=SignalPolynomial)
print(s.position_estimate)

print("Ordinary Least Squares")
s = OrdinaryLS([0, 1, 2], 2, SignalPolynomial)
s.solve()
print(s.position_estimate)

print("Alternating Least Squares")
s = AlternatingLS([0, 1, 2], 2, SignalPolynomial, hold_edges=False)
s.solve()
print(s.position_estimate)

print("Constrained Alternating Least Squares")
u = ConstrainedALS([0, 1, 2], 2, SurfacePolynomial, [1, 2, 3])
u.solve()
print(s.position_estimate)
