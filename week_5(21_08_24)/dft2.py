import numpy as np
import matplotlib.pyplot as plt

def DFT(X, N):
    x = []
    for k in np.arange(0, N):
        s = 0
        for n in np.arange(0, N):
            s += X[n] * np.exp(-1j * 2 * np.pi * n * k / N)
        x.append(s)
    return np.array(x)  

n = np.arange(0, 1800)
x = np.cos(2 * np.pi * 200 * (n / 1800))  
N = len(x)
re = DFT(x, N)  


5
plt.subplot(3, 1, 1)
plt.title("DFT of Signal")
plt.plot(re)

magnitude = np.abs(re)
angle = np.angle(re)

plt.subplot(3, 1, 2)
plt.title("Magnitude of Signal")
plt.plot(magnitude)

plt.subplot(3, 1, 3)
plt.title("Angle of Signal")
plt.plot(angle)

plt.tight_layout()
plt.show()
