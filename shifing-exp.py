from matplotlib import pylab
from pylab import rcParams
from signals import *
from scipy import optimize
rcParams['figure.figsize'] = 5, 4

signal = SignalExp([0, 1, 0])

change = np.array([0, 1, 0])
alpha = np.linspace(0,5)
t = np.linspace(0,2*np.pi)
sample_pos = np.multiply([0.3,0.5,0.7,0.9,1.1,1.3,1.5,1.7],np.pi)

pylab.ion()
pylab.plot(t,signal.get_samples(t),'k',lw=2)
sample_val = signal.get_samples(sample_pos)
pylab.stem(sample_pos, sample_val,'b')
pylab.ioff()
pylab.show()

pylab.ion()
for s in sample_pos:
    p = signal.path(s,0.1*change)
    pylab.plot(p,alpha,lw=3)
pylab.ylabel(r'$\alpha$',fontsize=30)
pylab.xlim([0,2*np.pi])
pylab.ioff()
pylab.show()

pylab.ion()
p1 = signal.path(sample_pos[0],0.1*change)
for s in sample_pos:
    p = signal.path(s,0.1*change)
    pylab.plot(p1,p,lw=3)
# pylab.xlim([0.6,1.4])
# pylab.ylim([0.6,2.8])
pylab.ioff()
pylab.show()
