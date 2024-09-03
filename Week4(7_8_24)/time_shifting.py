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

x1 = np.array([1, 2, 3, 4, 0, 0, 0, 0])
x2 = np.array([0, 0, 0, 0, 4, 3, 2, 1])

w, X1 = dtft(x1)
_, X2 = dtft(x2)

shift_x1 = 2
shift_x2 = -3

x1_shifted = np.roll(x1, shift_x1)
x2_shifted = np.roll(x2, shift_x2)

_, X1_shifted = dtft(x1_shifted)
_, X2_shifted = dtft(x2_shifted)

X1_theoretical_shifted = np.abs(X1) * np.exp(-1j * w * shift_x1)
X2_theoretical_shifted = np.abs(X2) * np.exp(-1j * w * shift_x2)

plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(w, np.abs(X1), label='DTFT of x1')
plt.title('DTFT of x1')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()

plt.subplot(2, 2, 2)
plt.plot(w, np.abs(X2), label='DTFT of x2')
plt.title('DTFT of x2')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.plot(w, np.abs(X1_shifted), label='DTFT of Time-Shifted x1')
plt.plot(w, np.abs(X1_theoretical_shifted), label='Theoretical DTFT of Time-Shifted x1', linestyle='--')
plt.title('Time-Shifted DTFT of x1')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.plot(w, np.abs(X2_shifted), label='DTFT of Time-Shifted x2')
plt.plot(w, np.abs(X2_theoretical_shifted), label='Theoretical DTFT of Time-Shifted x2', linestyle='--')
plt.title('Time-Shifted DTFT of x2')
plt.xlabel('Frequency (rad/sample)')
plt.ylabel('Magnitude')
plt.grid()

plt.tight_layout()
plt.show()