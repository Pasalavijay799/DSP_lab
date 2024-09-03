import numpy as np
import matplotlib.pyplot as plt

def dtft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])

X1 = dtft(x1)
X2 = dtft(x2)


a = 2
b = 3
x3 = a * x1 + b * x2
X3 = dtft(x3)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.stem(np.abs(X1))
plt.title("DTFT of x1")
plt.subplot(3, 1, 2)
plt.stem(np.abs(X2))
plt.title("DTFT of x2")
plt.subplot(3, 1, 3)
plt.stem(np.abs(a * X1 + b * X2))
plt.title("DTFT of a*x1 + b*x2")
plt.tight_layout()
plt.show()

if np.allclose(X3, a * X1 + b * X2):
    print("Linearity property holds")
else:
    print("Linearity property does not hold")