import numpy as np
import matplotlib.pyplot as plt

t=0.005
fs=8000
f=200
T=np.arange(0,t,1.0/fs)

x=np.sin(2*np.pi*f*T)

plt.stem(x)

plt.show()



