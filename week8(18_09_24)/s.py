import numpy as np
import matplotlib.pyplot as plt

# Generate a random signal
np.random.seed(0)
n = np.arange(0, 100, 1)
x = np.random.rand(100)

# Define the exponential signal
a = 0.9
exp_signal = a**n

# Multiply the random signal with the exponential signal
y = x * exp_signal

# Plot the original signal and the multiplied signal
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(n, x, label='Original Signal')
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(n, y, label='Multiplied Signal')
plt.title('Multiplied Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()