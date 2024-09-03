import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
def DFT(X,N):
    x=[]
    for k in np.arange(0,N):
        s=0
        for n in np.arange(0,N):
            s=s+X[n]*np.exp(-1j*2*np.pi*n*(k/N))
        x.append(s)
    return x
fs, x = wave.read('/home/vijay/Downloads/n.wave')
n=len(x)
re=DFT(x,n)
plt.title("DFT of signal")
plt.plot(re)
