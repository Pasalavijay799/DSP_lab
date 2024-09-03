import numpy as np
import matplotlib.pyplot as plt

def dtft(x):
    X = []
    w = np.arange(-np.pi, np.pi, 0.0001*np.pi)
    for f in w:
        s = 0
        for n in range(len(x)):
            s += x[n] * np.exp(-1j * f * n)
        X.append(s)
    return w, np.array(X)

x1 = np.array([1, 2, 3, 4])
x2 = np.array([4, 3, 2, 1])

w, X1 = dtft(x1)
_, X2 = dtft(x2)

X1_plus_X2 = X1 + X2
_, X3 = dtft(x1 + x2)

if np.allclose(X1_plus_X2, X3):
    print("Linearity property verified: X1 + X2 == DTFT(x1 + x2)")
else:
    print("Linearity property not verified")


plt.plot(w, np.abs(X1), label='DTFT of x1')
plt.plot(w, np.abs(X2), label='DTFT of x2')
plt.plot(w, np.abs(X1_plus_X2), label='X1 + X2', linestyle='--')
plt.plot(w, np.abs(X3), label='DTFT(x1 + x2)', linestyle=':')

plt.title('Linearity Property of DTFT')

plt.show()
