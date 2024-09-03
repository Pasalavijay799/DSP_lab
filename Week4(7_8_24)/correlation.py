import numpy as np
import matplotlib.pyplot as plt

def dtft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def correlation(x1, x2):
    N = len(x1)
    r = np.zeros(N)
    for k in range(N):
        for n in range(N):
            r[k] += x1[n] * x2[(n-k) % N]
    return r

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])

X1 = dtft(x1)
X2 = dtft(x2)


R = correlation(x1, x2)
R_dtft = dtft(R)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.stem(np.abs(R_dtft))
plt.title("DTFT of correlation of x1 and x2")
plt.subplot(2, 1, 2)
plt.stem(np.abs(X1 * np.conj(X2)))
plt.title("Product of DTFTs of x1 and x2")
plt.tight_layout()
plt.show()

if np.allclose(R_dtft, X1 * np.conj(X2)):
    print("Correlation property holds")
else:
    print("Correlation property does not hold")