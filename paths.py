import numpy as np
import pylab

# WARNING: parameters[0] is the coefficient of the highest power of x
first_parameters = np.array([0.0, -2.0, 1.0])
N = 3
steps = 10
step_size = 0.01
samples = [0]*N
for i in range(0, N):
    samples[i] = np.polyval(first_parameters, 1.0*i/N)

second_parameters = np.array([1.0, 3.0, 2.0])
a = np.polyval(second_parameters, 0)
b = np.polyval(second_parameters, 1)
second_parameters[-1] -= a
second_parameters[-2] -= (a+b)

for i in range(0, N):
    tmp_parameters = first_parameters
    tmp_parameters[-1] -= samples[i]
    z = np.roots(tmp_parameters)
    print(np.real(z))
    for k in range(0, steps):
        z = np.real(np.roots(tmp_parameters))
        tmp_parameters -= step_size*second_parameters
        pylab.plot(1.0*i/N, z[0], 'o')

pylab.show()
