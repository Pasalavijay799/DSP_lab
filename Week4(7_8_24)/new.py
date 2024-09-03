import numpy as np
import matplotlib.pyplot as plt


def dtft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-1j * np.pi * k * n / N)
    return X


def time_shift(X, shift):
    N = len(X)
    X_shifted = np.zeros(N, dtype=complex)
    for k in range(N):
        X_shifted[(k - shift) % N] = X[k]
    return X_shifted


x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])

X1 = dtft(x1)
X2 = dtft(x2)
X3 = dtft(x1 + x2)


if np.allclose(X1 + X2, X3):
    print("X1 + X2 == X3")
    
    shift = 2
    X1_shifted = time_shift(X1, shift)
    X2_shifted = time_shift(X2, shift)
    X3_shifted = time_shift(X3, shift)


    plt.figure(figsize=(12, 6))


    plt.subplot(3, 2, 1)
    plt.stem(np.abs(X1))
    plt.title("|X1|")

    plt.subplot(3, 2, 2)
    plt.stem(np.abs(X1_shifted))
    plt.title("|X1_shifted|")

    plt.subplot(3, 2, 3)
    plt.stem(np.abs(X2))
    plt.title("|X2|")

    plt.subplot(3, 2, 4)
    plt.stem(np.abs(X2_shifted))
    plt.title("|X2_shifted|")

    plt.subplot(3, 2, 5)
    plt.stem(np.abs(X3))
    plt.title("|X3|")

    plt.subplot(3, 2, 6)
    plt.stem(np.abs(X3_shifted))
    plt.title("|X3_shifted|")

    plt.tight_layout()
    plt.show()
else:
    print("X1 + X2 != X3")