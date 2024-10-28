
import numpy as np

def cycle_delay(x, m):
    N = len(x)
    y = []
    for n in range(0, N):
        idx = (n - m) % N
        y.append(x[idx])
    return y

def circularconv(x1, x2):
    z = []
    x2_padded = x2 + [0] * (len(x1) - len(x2))
    for n in range(len(x1)):
        y = cycle_delay(x2_padded, n)
        z.append(np.dot(x1, y))
    return z

x1 = [4, 3, 2, 1]
x2 = [1, -2, 0, 4]
Y = circularconv(x1, x2)
print(Y)