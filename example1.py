from solvers import *
from plots import *


s = Solver([0, 1, 2], 2, SignalPolynomial)
print(s.position_estimate)

s = OrdinaryLS([0, 1, 2], 2, SignalPolynomial)
s.solve()
print(s.position_estimate)

s = AlternatingLS([0, 1, 2], 2, SignalPolynomial, hold_edges=False)
s.solve()
print(s.position_estimate)
print(s.hold_edges)

try:
    s = ConstrainedALS([0, 1, 2], 2, ConstrainedPolynomial, [1, 2, 3])
    s.solve()
    print(s.position_estimate)
except NotImplementedError:
    print("ConstrainedPolynomial is an abstract class!")

u = ConstrainedALS([0, 1, 2], 2, SurfacePolynomial, [1, 2, 3])
u.solve()
