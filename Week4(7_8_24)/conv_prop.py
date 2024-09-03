

import numpy as np
import matplotlib.pyplot as plt

def dtft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * np.pi * k * n / N)
    return X

def convolution(x1, x2):
    N = len(x1) + len(x2) - 1
    x1_zero_padded = np.pad(x1, (0, N - len(x1)))
    x2_zero_padded = np.pad(x2, (0, N - len(x2)))
    return np.convolve(x1_zero_padded, x2_zero_padded, mode='full')[:N]

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])

X1 = dtft(x1)
X2 = dtft(x2)

x1_conv_x2 = convolution(x1, x2)
X1_conv_X2 = dtft(x1_conv_x2)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.stem(np.abs(X1))
plt.title("|X1|")
plt.subplot(2, 2, 2)
plt.stem(np.abs(X2))
plt.title("|X2|")
plt.subplot(2, 2, 3)
plt.stem(np.abs(X1_conv_X2))
plt.title("|X1_conv_X2|")
plt.subplot(2, 2, 4)
plt.stem(np.abs(np.pad(X1 * X2, (0, 3))))
plt.title("|X1 * X2|")
plt.tight_layout()
plt.show()

if np.allclose(np.pad(X1 * X2, (0, 3)), X1_conv_X2):
    print("Convolution property holds")
else:
    print("Convolution property does not hold")