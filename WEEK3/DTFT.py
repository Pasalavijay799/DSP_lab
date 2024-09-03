

import numpy as np
from matplotlib import pyplot as plt


def dtft(x_n,omega):

    n = np.arange(len(x_n))
    
    n,omega = np.meshgrid(n,omega,indexing="ij")
    x_w = np.sum(x_n[n]*np.exp(-1j*omega*n),axis=0)

    return x_w

def plot_dtft(x_n):

    omega = np.arange(-np.pi,np.pi,0.0001*np.pi)

    x_w = dtft(x_n,omega)

    plt.subplot(2,1,1)

    plt.plot(omega, abs(x_w))
    plt.title("Magnitude of DTFT")
    plt.xlabel("Frequence(w)")
    plt.ylabel("|x(w)|")

    plt.subplot(2,1,2)
    plt.plot(omega, np.angle(x_w))
    plt.title("phase of DTFT")
    plt.xlabel("Frequence(w)")
    plt.ylabel("<x(w)")
    
    #plt.tight_layout()
    plt.show()


x_n = np.array([1,2,3,4])

plot_dtft(x_n)

    







