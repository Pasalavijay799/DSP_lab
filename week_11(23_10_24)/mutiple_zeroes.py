import numpy as np
import matplotlib.pyplot as plt

num_zeros = int(input("Enter the number of zeros: "))
zeros = []

for i in range(num_zeros):
    zero_input = input(f"Enter zero {i + 1} (e.g., 1+2j): ")
    zeros.append(complex(zero_input))

w = np.linspace(-10, 10, 1000)
H_w = np.prod([(1 - z / 1j * w) for z in zeros], axis=0)

magnitude_H = np.abs(H_w)
phase_H = np.angle(H_w)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(w, magnitude_H, 'b')
plt.title('Magnitude of H(w)')
plt.xlabel('Frequency (w)')
plt.ylabel('|H(w)|')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(w, phase_H, 'r')
plt.title('Phase of H(w)')
plt.xlabel('Frequency (w)')
plt.ylabel('Phase of H(w) (radians)')
plt.grid()

plt.tight_layout()
plt.show()