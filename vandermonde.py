import numpy as np
import numpy.linalg as la
from matplotlib import pylab


def generalized_vandermonde(n, alpha, *args):
    if args is ():
        x = np.array(list(range(1, 2*n)))
    else:
        assert len(args[0][0]) is 2*n-1
        x = np.array(args[0][0])  #czemu tak? Czy to nie powinno by? si? da? zrobi? lepiej?
    v1 = np.vander(np.exp(1j*x), n)
    v2 = np.vander(np.exp(1j*alpha*x), n)
    v = np.hstack((v1, v2))
    v = v[:, 0:-1]
    v = np.real(v)
    return np.abs(la.det(v))


n = 6
alpha = np.linspace(1,2,10000)
determinants = []
# vec = np.array([[0,0.5,0.9,1.1,1.5,2,2.5,3,3.3]])
# vec = np.array([[ 1.83739915, 2.43379158,  0.0102245,  5.06905912,  2.94455721,  4.32672297,
                   # 4.14166927,  5.39895411,  0.38197981]])
vec = np.random.rand(1,2*n-1)*np.pi*2
print(vec)

for a in alpha:
    determinants.append(generalized_vandermonde(n,a,vec))

pylab.semilogy(alpha,np.abs(determinants),'g',linewidth=2)
pylab.xlabel(r'$\alpha$')
pylab.ylabel(r'$\det(V)$')
# pylab.ylim(1e-11,1e3)
pylab.show()