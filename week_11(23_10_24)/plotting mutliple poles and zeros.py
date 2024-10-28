import numpy as np 
from matplotlib import pyplot as plt
from scipy import signal
#zeros
w1=0
w2=0.1
w3=-0.1
z1=1.5*np.exp(1j*w1)
z2=0.9*np.exp(1j*w2)
z3=0.9*np.exp(1j*w3)
#poles
a1=np.pi
a2=2.927
a3=-2.927
p1=0.8*np.exp(1j*a1)
p2=0.7*np.exp(1j*a2)
p3=0.7*np.exp(1j*a3)
w,h=signal.freqz_zpk([z1,z2,z3],[p1,p2,p3],1)
plt.stem(w/np.pi,20*np.log(abs(h)))
plt.show()
