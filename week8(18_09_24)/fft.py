import numpy as np
x = [1, -1, 2, 4]

def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        e = x[0::2]
        o = x[1::2]
        E = fft(e)
        O = fft(o)
        w = np.exp(-1j * 2 * np.pi * np.arange(n // 2) / n)
        return np.concatenate([E + w * O, E - w * O])

print(fft(x))
