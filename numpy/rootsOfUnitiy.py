from matplotlib.pyplot import *
from numpy import * 

N = 10

unity_roots = array([exp(1j*2*pi*k/N) for k in range (N) ])

axes(aspect='equal')
plot(unity_roots.real, unity_roots.imag, 'o')
allclose(unity_roots**N, 1)