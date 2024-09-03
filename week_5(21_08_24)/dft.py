import numpy as np
import matplotlib.pyplot as plt
def DFT(X,N):
    x=[]
    for k in np.arange(0,N):
        s=0
        for n in np.arange(0,N):
            s=s+X[n]*np.exp(-1j*2*np.pi*n*(k/N))
        x.append(s)
    return x
x=[1,2,3,4,5]
n=len(x)
re=DFT(x,n)
plt.subplot(3,1,1)
plt.title("DFT of signal")
plt.plot(re)
magnitude=np.abs(re)
plt.subplot(3,1,2)
plt.title("magnitude  of signal")
plt.plot(magnitude)
plt.subplot(3,1,3)
angle=np.angle(re)
plt.title("angle of signal")
plt.plot(angle)
plt.tight_layout()
plt.show()

