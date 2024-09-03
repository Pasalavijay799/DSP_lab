import numpy as np
import matplotlib.pyplot as plt

def dtft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * np.pi * k * n / N)
    return X

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])

X1 = dtft(x1)
X2 = dtft(x2)


E1 = np.abs(X1)**2 / len(x1)
E2 = np.abs(X2)**2 / len(x2)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.stem(E1)
plt.title("Energy Spectral Density of x1")
plt.subplot(2, 1, 2)
plt.stem(E2)
plt.title("Energy Spectral Density of x2")
plt.tight_layout()
plt.show()

if np.allclose(E1, E2):
    print("Energy spectral density property holds")
else:
    print("Energy spectral density property does not hold")