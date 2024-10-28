import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, ])
h = np.array([4,5,6])


def linear_convolution(x, h):
    N = len(x)
    M = len(h)
    conv_result = np.zeros(N + M - 1)
    for n in range(N + M - 1):
        for k in range(N):
            if n - k >= 0 and n - k < M:
                conv_result[n] += x[k] * h[n - k]
    return conv_result


def DFT(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return X


def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N


linear_conv = linear_convolution(x, h)

N = len(x) + len(h) - 1
x_padded = np.pad(x, (0, N - len(x)))
h_padded = np.pad(h, (0, N - len(h)))


X_padded = DFT(x_padded)
H_padded = DFT(h_padded)
Y_padded = X_padded * H_padded
circular_conv = IDFT(Y_padded)


plt.subplot(1, 2, 1)
plt.stem(linear_conv)
plt.title('Linear Convolution')

plt.subplot(1, 2, 2)
plt.stem(np.real(circular_conv))
plt.title('Circular Convolution')
print(Y_padded)
plt.tight_layout()
plt.show()